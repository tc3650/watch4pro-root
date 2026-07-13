#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#define KGSL_IOC_TYPE 0x09

struct kgsl_map_user_mem { int fd; unsigned long gpuaddr; size_t len; size_t offset;
    unsigned long hostptr; unsigned int memtype; unsigned int flags; };
struct kgsl_drawctxt_create { unsigned int flags; unsigned int drawctxt_id; };
struct kgsl_drawctxt_destroy { unsigned int drawctxt_id; };
struct kgsl_gpu_command {
    uint64_t flags; uint64_t cmdlist; unsigned int cmdsize; unsigned int numcmds;
    uint64_t objlist; unsigned int objsize; unsigned int numobjs;
    uint64_t synclist; unsigned int syncsize; unsigned int numsyncs;
    unsigned int context_id; unsigned int timestamp;
};
struct kgsl_command_object {
    uint64_t offset; uint64_t gpuaddr; uint64_t size;
    unsigned int flags; unsigned int id;
};

#define KGSL_USER_MEM_TYPE_ADDR 0x00000002
#define KGSL_CMDLIST_IB 0x00000001U
#define KGSL_COMMAND_SYNC 0x1

#define IOCTL_KGSL_DRAWCTXT_CREATE _IOWR(KGSL_IOC_TYPE, 0x13, struct kgsl_drawctxt_create)
#define IOCTL_KGSL_DRAWCTXT_DESTROY _IOW(KGSL_IOC_TYPE, 0x14, struct kgsl_drawctxt_destroy)
#define IOCTL_KGSL_MAP_USER_MEM _IOWR(KGSL_IOC_TYPE, 0x15, struct kgsl_map_user_mem)
#define IOCTL_KGSL_GPU_COMMAND _IOWR(KGSL_IOC_TYPE, 0x4A, struct kgsl_gpu_command)

#define CP_TYPE7_PKT(op, cnt) ((7<<28)|((op)<<16)|((cnt)&0x3FFF))
#define CP_MEM_WRITE 0x3d
#define CP_NOP 0x10
#define CP_WAIT_FOR_IDLE 0x26
#define CP_WAIT_MEM_WRITES 0x12

#define upper_32_bits(n) ((uint32_t)(((n)>>16)>>16))
#define lower_32_bits(n) ((uint32_t)(n))

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open\n"); return 1; }
    printf("[+] fd=%d\n", fd);

    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) {
        printf("[-] ctx: %s\n", strerror(errno)); close(fd); return 1;
    }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* Buffer: commands at 0, result at 0x1000 */
    uint32_t *buf = mmap(NULL, 0x2000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(buf, 0xAB, 0x2000);
    volatile uint32_t *res = buf + 0x1000/4;

    /* Build commands: NOP, WAIT_FOR_IDLE, MEM_WRITE(0xCAFE to res), WAIT_MEM_WRITES */
    uint32_t *cmds = buf;
    *cmds++ = CP_TYPE7_PKT(CP_NOP, 0);
    *cmds++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
    *cmds++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    /* GPU address of result - will fill after mapping */
    uint32_t *addr_lo = cmds; cmds++;
    uint32_t *addr_hi = cmds; cmds++;
    *cmds++ = 0xCAFEBABE;  /* value */
    *cmds++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
    *cmds++ = CP_TYPE7_PKT(CP_NOP, 0);
    int cmd_dwords = cmds - buf;
    int cmd_bytes = cmd_dwords * 4;
    int sc_bytes = 0; /* will be set in Approach 4 */
    printf("[+] cmd: %d dwords, %d bytes\n", cmd_dwords, cmd_bytes);

    /* Map to GPU - try mapping just the command portion if full buffer fails */
    uint64_t base_ga = 0;
    struct kgsl_map_user_mem m = { .len = 0x2000, .hostptr = (unsigned long)buf,
                                    .memtype = KGSL_USER_MEM_TYPE_ADDR };
    int r = ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m);
    if (r) { printf("[-] map: %s\n", strerror(errno)); goto end; }
    base_ga = m.gpuaddr;
    printf("[+] buf: cpu=%p gpu=0x%lx\n", buf, (unsigned long)base_ga);

    uint64_t res_ga = base_ga + 0x1000;
    *addr_lo = lower_32_bits(res_ga);
    *addr_hi = upper_32_bits(res_ga);
    __builtin___clear_cache((char*)buf, (char*)buf+0x2000);

    /* Approach 1: normal submit */
    printf("\n== Approach 1: normal submit ==\n");
    {
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .cmdlist = (uint64_t)(unsigned long)&obj,
            .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 1 };
        struct kgsl_gpu_command *hreq = malloc(sizeof(*hreq));
        memcpy(hreq, &req, sizeof(req));
        r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, hreq);
        printf("submit: %s ts=%u flags=0x%lx\n", r ? strerror(errno) : "OK",
               hreq->timestamp, (unsigned long)hreq->flags);
        free(hreq);
    }
    usleep(500000);
    printf("res[0]=0x%08x\n", res[0]);

    /* Approach 2: with KGSL_COMMAND_SYNC flag, heap cmdlist */
    printf("\n== Approach 2: SYNC flag + heap ==\n");
    if (res[0] != 0xCAFEBABE) {
        struct kgsl_command_object *hobj = malloc(sizeof(*hobj));
        memset(hobj, 0, sizeof(*hobj));
        hobj->gpuaddr = base_ga; hobj->size = cmd_bytes; hobj->flags = KGSL_CMDLIST_IB;
        
        struct kgsl_gpu_command *hreq = calloc(1, sizeof(*hreq));
        hreq->flags = KGSL_COMMAND_SYNC;
        hreq->cmdlist = (uint64_t)(unsigned long)hobj;
        hreq->cmdsize = sizeof(*hobj);
        hreq->numcmds = 1;
        hreq->context_id = ctx.drawctxt_id;
        hreq->timestamp = 1;
        
        printf("[*] submitting (heap+sync)...\n");
        r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, hreq);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", hreq->timestamp);
        usleep(500000);
        printf("res[0]=0x%08x\n", res[0]);
        free(hobj); free(hreq);
    }

    /* Approach 3: multiple IBs */
    printf("\n== Approach 3: multiple IBs ==\n");
    if (res[0] != 0xCAFEBABE) {
        struct kgsl_command_object objs[3];
        memset(objs, 0, sizeof(objs));
        for (int i = 0; i < 3; i++) {
            objs[i].gpuaddr = base_ga;
            objs[i].size = cmd_bytes;
            objs[i].flags = KGSL_CMDLIST_IB;
        }
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)objs, .cmdsize = sizeof(*objs), .numcmds = 3,
            .context_id = ctx.drawctxt_id, .timestamp = 1 };
        printf("[*] submitting 3 IBs...\n");
        r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
        usleep(500000);
        printf("res[0]=0x%08x\n", res[0]);
    }

    /* Approach 4: try without mapping as IB - just submit raw */
    printf("\n== Approach 4: short IB ==\n");
    if (res[0] != 0xCAFEBABE) {
        /* Try with just the MEM_WRITE part (no NOP/WAIT prefix) */
        uint32_t *short_cmds = cmds;
        /* Build: MEM_WRITE(0xDEADBEEF to res_ga) */
        /* Re-write commands at 0x100 in the buffer (keep result at 0x1000) */
        uint32_t *sc = buf + 0x100/4;
        *sc++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
        *sc++ = lower_32_bits(res_ga);
        *sc++ = upper_32_bits(res_ga);
        *sc++ = 0xDEADBEEF;
        *sc++ = CP_TYPE7_PKT(CP_NOP, 0);
        sc_bytes = ((char*)sc - (char*)(buf + 0x100/4));
        
        __builtin___clear_cache((char*)buf, (char*)buf+0x2000);
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga + 0x100,
            .size = sc_bytes, .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj),
            .numcmds = 1, .context_id = ctx.drawctxt_id };
        printf("[*] submitting short IB at offset 0x100...\n");
        r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
        usleep(500000);
        printf("res[0]=0x%08x res[64]=0x%08x\n", res[0], *(buf + 0x100/4 + 3));
    }

    /* Approach 5: try with different context flags */
    printf("\n== Approach 5: different context flags ==\n");
    if (res[0] != 0xCAFEBABE && res[0] != 0xDEADBEEF) {
        /* Try creating a context with NO_USER_GENERATED_TS */
        uint32_t test_flags[] = { 0x00000000, 0x00000010, 0x00001810, 0x00003810 };
        for (int f = 0; f < 4; f++) {
            struct kgsl_drawctxt_create tctx = { .flags = test_flags[f] };
            if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &tctx)) continue;
            printf("[*] testing ctx flags=0x%08x id=%u\n", test_flags[f], tctx.drawctxt_id);
            
            struct kgsl_command_object obj = { .gpuaddr = base_ga + 0x100,
                .size = sc_bytes, .flags = KGSL_CMDLIST_IB };
            struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
                .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj),
                .numcmds = 1, .context_id = tctx.drawctxt_id };
            r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
            printf("    submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
            usleep(300000);
            printf("    res[0]=0x%08x\n", res[0]);
            
            if (res[0] == 0xDEADBEEF) {
                printf("✅ ctx flags 0x%08x WORKS!\n", test_flags[f]);
                break;
            }
            { struct kgsl_drawctxt_destroy d = { .drawctxt_id = tctx.drawctxt_id };
              ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
        }
    }
    if (res[0] == 0xCAFEBABE || res[0] == 0xDEADBEEF)
        printf("✅ GPU WORKS!\n");
    else
        printf("❌ GPU not responding\n");

end:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    return 0;
}