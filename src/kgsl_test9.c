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
#define CP_MEM_TO_MEM 0x73
#define CP_MEM_WRITE 0x3d

#define u32(x) ((uint32_t)(x))
#define lo32(x) ((uint32_t)((x)&0xFFFFFFFF))
#define hi32(x) ((uint32_t)(((x)>>16)>>16))

/* Use mprotect to force cache coherency (no DC CIVAC - blocked by kernel) */
static inline void cache_sync(void *addr, int len) {
    void *aligned = (void*)((uintptr_t)addr & ~0xFFF);
    size_t aligned_len = ((uintptr_t)addr + len + 0xFFF) & ~0xFFF;
    aligned_len -= (uintptr_t)aligned;
    mprotect(aligned, aligned_len, PROT_NONE);
    mprotect(aligned, aligned_len, PROT_READ|PROT_WRITE);
}

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open\n"); return 1; }
    printf("[+] fd=%d\n", fd);

    /* Create context */
    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) { printf("[-] ctx\n"); close(fd); return 1; }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* Buffer: commands at 0, src data at 0x800, dst result at 0x1000 */
    uint32_t *buf = mmap(NULL, 0x2000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(buf, 0, 0x2000);

    /* Pre-fill source data */
    volatile uint32_t *src = buf + 0x800/4;
    src[0] = 0x12345678;
    src[1] = 0x9ABCDEF0;

    /* Pre-fill dst with marker */
    volatile uint32_t *dst = buf + 0x1000/4;
    dst[0] = 0xDEAD;
    dst[1] = 0xBEEF;

    /* Map all to GPU */
    uint64_t base_ga = 0;
    {
        struct kgsl_map_user_mem m = { .len = 0x2000, .hostptr = (unsigned long)buf,
                                        .memtype = KGSL_USER_MEM_TYPE_ADDR };
        if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) {
            printf("[-] map: %s\n", strerror(errno)); goto end;
        }
        base_ga = m.gpuaddr;
    }
    printf("[+] mapped: cpu=%p gpu=0x%lx\n", buf, (unsigned long)base_ga);

    uint64_t src_ga = base_ga + 0x800;
    uint64_t dst_ga = base_ga + 0x1000;

    /* === Test 1: CP_MEM_TO_MEM (GPU copy) === */
    printf("\n== Test 1: CP_MEM_TO_MEM ==\n");
    {
        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        /* CP_MEM_TO_MEM: 5 dwords: flags, dst_lo, dst_hi, src_lo, src_hi */
        *p++ = CP_TYPE7_PKT(CP_MEM_TO_MEM, 5);
        *p++ = 0; /* flags */
        *p++ = lo32(dst_ga);     /* dst */
        *p++ = hi32(dst_ga);
        *p++ = lo32(src_ga);     /* src */
        *p++ = hi32(src_ga);
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;

        __builtin___clear_cache((char*)buf, (char*)buf + 0x2000);

        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj),
            .numcmds = 1, .context_id = ctx.drawctxt_id, .timestamp = 1 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
        
        if (!r) {
            /* Invalidate data cache before reading */
            cache_sync((void*)dst, 8);
            printf("  src[0]=0x%08x src[1]=0x%08x\n", src[0], src[1]);
            printf("  dst[0]=0x%08x dst[1]=0x%08x\n", dst[0], dst[1]);
            if (dst[0] == 0x12345678)
                printf("✅✅✅ CP_MEM_TO_MEM WORKS!\n");
            else if (dst[0] == 0xDEAD)
                printf("❌ GPU didn't copy\n");
            else
                printf("⚠️  dst changed to unexpected: 0x%08x\n", dst[0]);
        }
    }

    /* === Test 2: CP_MEM_WRITE with known-good addresses === */
    printf("\n== Test 2: CP_MEM_WRITE (direct write) ==\n");
    {
        dst[0] = 0xBBBBBBBB;
        __builtin___clear_cache((char*)buf, (char*)buf + 0x2000);

        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        /* MEM_WRITE: count=3: addr_lo, addr_hi, value */
        *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
        *p++ = lo32(dst_ga);
        *p++ = hi32(dst_ga);
        *p++ = 0xDEADBEEF;
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;

        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj),
            .numcmds = 1, .context_id = ctx.drawctxt_id, .timestamp = 2 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
        
        if (!r) {
            cache_sync((void*)dst, 4);
            printf("  dst[0]=0x%08x\n", dst[0]);
            if (dst[0] == 0xDEADBEEF)
                printf("✅✅✅ CP_MEM_WRITE WORKS!\n");
            else
                printf("❌ dst[0]=0x%08x (expected 0xDEADBEEF)\n", dst[0]);
        }
    }

    /* === Test 3: Poll dst for 10 seconds === */
    if (dst[0] != 0x12345678 && dst[0] != 0xDEADBEEF) {
        printf("\n== Test 3: Long poll (10s) ==\n");
        /* Submit MEM_TO_MEM again */
        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        *p++ = CP_TYPE7_PKT(CP_MEM_TO_MEM, 5);
        *p++ = 0; *p++ = lo32(dst_ga); *p++ = hi32(dst_ga);
        *p++ = lo32(src_ga); *p++ = hi32(src_ga);
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj),
            .numcmds = 1, .context_id = ctx.drawctxt_id, .timestamp = 3 };
        ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        
        printf("[*] polling dst for 10 seconds (DC CIVAC every 500ms)...\n");
        for (int i = 0; i < 20; i++) {
            cache_sync((void*)dst, 8);
            if (dst[0] == 0x12345678) {
                printf("  ✅ Found after %d polls!\n", i);
                break;
            }
            printf("  [%d] dst=0x%08x\n", i, dst[0]);
            usleep(500000);
        }
    }

end:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    return 0;
}