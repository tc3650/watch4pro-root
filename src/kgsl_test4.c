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

struct kgsl_map_user_mem {
    int fd;
    unsigned long gpuaddr;
    size_t len;
    size_t offset;
    unsigned long hostptr;
    unsigned int memtype;
    unsigned int flags;
};

struct kgsl_drawctxt_create {
    unsigned int flags;
    unsigned int drawctxt_id;
};

struct kgsl_drawctxt_destroy {
    unsigned int drawctxt_id;
};

struct kgsl_gpu_command {
    uint64_t flags;
    uint64_t cmdlist;
    unsigned int cmdsize;
    unsigned int numcmds;
    uint64_t objlist;
    unsigned int objsize;
    unsigned int numobjs;
    uint64_t synclist;
    unsigned int syncsize;
    unsigned int numsyncs;
    unsigned int context_id;
    unsigned int timestamp;
};

struct kgsl_command_object {
    uint64_t offset;
    uint64_t gpuaddr;
    uint64_t size;
    unsigned int flags;
    unsigned int id;
};

#define KGSL_USER_MEM_TYPE_ADDR 0x00000002
#define KGSL_CMDLIST_IB 0x00000001U

#define IOCTL_KGSL_DRAWCTXT_CREATE   _IOWR(KGSL_IOC_TYPE, 0x13, struct kgsl_drawctxt_create)
#define IOCTL_KGSL_DRAWCTXT_DESTROY   _IOW(KGSL_IOC_TYPE, 0x14, struct kgsl_drawctxt_destroy)
#define IOCTL_KGSL_MAP_USER_MEM       _IOWR(KGSL_IOC_TYPE, 0x15, struct kgsl_map_user_mem)
#define IOCTL_KGSL_GPU_COMMAND        _IOWR(KGSL_IOC_TYPE, 0x4A, struct kgsl_gpu_command)

/* A7xx GPU packets */
#define CP_TYPE7_PKT(opcode, count) ((7 << 28) | ((opcode) << 16) | ((count) & 0x3FFF))
#define CP_NOP               0x10
#define CP_WAIT_FOR_IDLE     0x26
#define CP_WAIT_MEM_WRITES   0x12
#define CP_MEM_WRITE         0x3d
#define CP_SET_DRAW_STATE    0x43
#define CP_SET_MODE          0x45

#define upper_32_bits(n) ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n) ((uint32_t)(n))

static inline void add_gpuaddr(uint32_t **pp, uint64_t addr) {
    *(*pp)++ = lower_32_bits(addr);
    *(*pp)++ = upper_32_bits(addr);
}

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open: %s\n", strerror(errno)); return 1; }
    printf("[+] fd=%d\n", fd);

    /* Create context */
    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) {
        printf("[-] ctx: %s\n", strerror(errno)); close(fd); return 1;
    }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* Single buffer: draw_state at 0x000, payload at 0x1000, result at 0x2000 */
    uint32_t *buf = mmap(NULL, 0x4000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { printf("[-] mmap\n"); goto destroy; }
    memset(buf, 0xAB, 0x4000);

    /* Map to GPU */
    uint64_t base_ga = 0;
    {
        struct kgsl_map_user_mem m = { .len = 0x4000, .hostptr = (unsigned long)buf, 
                                        .memtype = KGSL_USER_MEM_TYPE_ADDR };
        if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) { printf("[-] map\n"); goto destroy; }
        base_ga = m.gpuaddr;
    }
    printf("[+] base: cpu=%p gpu=0x%lx\n", buf, (unsigned long)base_ga);

    uint64_t draw_ga = base_ga + 0x0000;
    uint64_t pl_ga   = base_ga + 0x1000;
    uint64_t res_ga  = base_ga + 0x2000;

    /* === Build draw state at offset 0x0000 === */
    uint32_t *dp = buf;
    /* MEM_WRITE: write 0xCAFEBABE to res_ga */
    *(dp++) = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    add_gpuaddr(&dp, res_ga);
    *(dp++) = 0xCAFEBABE;
    /* MEM_WRITE: write 0xDEADBEEF to res_ga + 4 */
    *(dp++) = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    add_gpuaddr(&dp, res_ga + 4);
    *(dp++) = 0xDEADBEEF;
    /* WAIT_MEM_WRITES */
    *(dp++) = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
    *(dp++) = CP_TYPE7_PKT(CP_NOP, 0);
    uint32_t draw_size = (char*)dp - (char*)buf; /* bytes */
    printf("[+] drawstate: ga=0x%lx size=%u\n", (unsigned long)draw_ga, draw_size);

    /* === Build payload at offset 0x1000 === */
    uint32_t *pp = buf + 0x1000/4;
    uint32_t *payload_start = pp;

    /* SET_MODE: immediate execution */
    *(pp++) = CP_TYPE7_PKT(CP_SET_MODE, 1);
    *(pp++) = 1; /* ALWAYS / immediate */
    
    /* SET_DRAW_STATE: point to draw state */
    *(pp++) = CP_TYPE7_PKT(CP_SET_DRAW_STATE, 3);
    add_gpuaddr(&pp, draw_ga);
    *(pp++) = draw_size; /* size in bytes? dwords? */

    *(pp++) = CP_TYPE7_PKT(CP_NOP, 0);
    uint32_t payload_size = (char*)pp - (char*)payload_start;

    printf("[+] payload: ga=0x%lx size=%u\n", (unsigned long)pl_ga, payload_size);

    /* Sync cache */
    __builtin___clear_cache((char*)buf, (char*)buf + 0x4000);

    /* Submit the payload IB via KGSL_GPU_COMMAND */
    {
        struct kgsl_command_object obj = {
            .gpuaddr = pl_ga,  /* GPU virtual address of payload */
            .size = payload_size,
            .flags = KGSL_CMDLIST_IB,
        };
        struct kgsl_gpu_command req = {
            .cmdlist = (uint64_t)(unsigned long)&obj,
            .cmdsize = sizeof(obj),
            .numcmds = 1,
            .context_id = ctx.drawctxt_id,
            .timestamp = 42,
        };
        printf("[*] submitting GPU command...\n");
        if (ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req)) {
            printf("[-] GPU_COMMAND: %s\n", strerror(errno));
            goto destroy;
        }
        printf("[+] submitted ts=%u\n", req.timestamp);
    }

    /* Poll for completion */
    volatile uint32_t *r = (volatile uint32_t *)(buf + 0x2000/4);
    printf("[*] polling...\n");
    for (int i = 0; i < 30; i++) {
        if (r[0] == 0xCAFEBABE) {
            printf("[%d] ✅ marker=0x%08x target=0x%08x\n", i, r[0], r[1]);
            if (r[1] == 0xDEADBEEF)
                printf("✅✅ GPU MEM_WRITE WORKS! Both values confirmed!\n");
            else
                printf("⚠️  marker OK but target=0x%08x\n", r[1]);
            goto destroy;
        }
        usleep(100000);
        if (i % 5 == 4) printf("[%d] marker=0x%08x\n", i, r[0]);
    }
    printf("❌ No GPU response after 3s\n");

destroy:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    printf("[+] DONE\n");
    return 0;
}