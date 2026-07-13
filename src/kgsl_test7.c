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
#define CP_NOP 0x10
#define CP_WAIT_FOR_IDLE 0x26

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open\n"); return 1; }
    printf("[+] fd=%d\n", fd);

    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) {
        printf("[-] ctx: %s\n", strerror(errno)); close(fd); return 1;
    }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* Buffer for commands only - no MEM_WRITE target */
    uint32_t *buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(buf, 0, 4096);

    /* Just NOPs and WAIT_FOR_IDLE - no memory write */
    uint32_t *p = buf;
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
    *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
    int cmd_bytes = (char*)p - (char*)buf;
    printf("[+] cmd: %d bytes\n", cmd_bytes);

    /* Map to GPU */
    uint64_t base_ga = 0;
    struct kgsl_map_user_mem m = { .len = 4096, .hostptr = (unsigned long)buf,
                                    .memtype = KGSL_USER_MEM_TYPE_ADDR };
    if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) {
        printf("[-] map: %s\n", strerror(errno)); goto end;
    }
    base_ga = m.gpuaddr;
    printf("[+] mapped: cpu=%p gpu=0x%lx\n", buf, (unsigned long)base_ga);

    __builtin___clear_cache((char*)buf, (char*)buf+4096);

    /* Test 1: Submit WITHOUT SYNC */
    printf("\n--- Test 1: NOP only, no sync ---\n");
    {
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .cmdlist = (uint64_t)(unsigned long)&obj,
            .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 1 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
    }

    /* Test 2: Submit WITH SYNC */
    printf("\n--- Test 2: NOP only, with sync ---\n");
    {
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj,
            .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 2 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
    }

    /* Test 3: With objlist (memory object) */
    printf("\n--- Test 3: with objlist ---\n");
    {
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        /* Also add our buffer as a memory object */
        struct kgsl_command_object memobj = { .gpuaddr = base_ga, .size = 4096,
                                               .flags = 0, .id = 0 };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .objlist = (uint64_t)(unsigned long)&memobj, .objsize = sizeof(memobj), .numobjs = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 3 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
    }

    /* Test 4: Try different command sizes */
    printf("\n--- Test 4: varying sizes ---\n");
    int sizes[] = { 8, 16, 32, 64, 128, 256 };
    for (int i = 0; i < 6; i++) {
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = sizes[i],
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 10+i };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("  size=%d: %s ts=%u\n", sizes[i], r ? strerror(errno) : "OK", req.timestamp);
    }

    /* Test 5: Without USER_GENERATED_TS flag */
    printf("\n--- Test 5: different ctx flags ---\n");
    {
        struct kgsl_drawctxt_create tctx = { .flags = 0x00000000 };
        if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &tctx) == 0) {
            printf("[*] new ctx id=%u\n", tctx.drawctxt_id);
            struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                                .flags = KGSL_CMDLIST_IB };
            struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
                .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
                .context_id = tctx.drawctxt_id, .timestamp = 50 };
            int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
            printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
            { struct kgsl_drawctxt_destroy d = { .drawctxt_id = tctx.drawctxt_id };
              ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
        }
    }

end:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    printf("[+] DONE\n");
    return 0;
}