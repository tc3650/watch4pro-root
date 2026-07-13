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
#define CP_WAIT_MEM_WRITES 0x12
#define CP_MEM_WRITE 0x3d

#define upper_32_bits(n) ((uint32_t)(((n)>>16)>>16))
#define lower_32_bits(n) ((uint32_t)(n))

/* Force cache coherency via mprotect trick, then try DC CIVAC inline asm */
static inline void cache_invalidate(void *addr, size_t len) {
    /* First, try mprotect approach (always safe) */
    void *aligned = (void*)((uintptr_t)addr & ~0xFFF);
    size_t aligned_len = ((uintptr_t)addr + len + 0xFFF) & ~0xFFF;
    aligned_len -= (uintptr_t)aligned;
    mprotect(aligned, aligned_len, PROT_NONE);
    mprotect(aligned, aligned_len, PROT_READ|PROT_WRITE);
    
    /* Also try DC CIVAC on each cache line (from Mesa/Freedreno) */
    /* dc civac = Clean and Invalidate data cache line by Virtual Address */
    char *start = (char*)((uintptr_t)addr & ~31);
    char *end = (char*)((uintptr_t)(addr + len + 31) & ~31);
    for (char *p = start; p < end; p += 32) {
        __asm volatile("mcr p15, 0, %0, c7, c14, 1" : : "r"(p) : "memory");
    }
    __asm volatile("dsb sy" : : : "memory");
}

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

    /* Build commands: WAIT_FOR_IDLE, MEM_WRITE(0xCAFEBABE to res), WAIT_MEM_WRITES, NOP */
    uint32_t *p = buf;
    *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
    *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    uint32_t *addr_lo = p; p++;
    uint32_t *addr_hi = p; p++;
    *p++ = 0xCAFEBABE; /* value */
    *p++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
    int cmd_bytes = (char*)p - (char*)buf;
    printf("[+] cmd: %d bytes\n", cmd_bytes);

    /* Map to GPU */
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

    /* Fill in the GPU address of the result */
    uint64_t res_ga = base_ga + 0x1000;
    *addr_lo = lower_32_bits(res_ga);
    *addr_hi = upper_32_bits(res_ga);

    /* Sync command buffer to GPU */
    __builtin___clear_cache((char*)buf, (char*)buf + 0x2000);

    /* Submit with SYNC */
    printf("[*] submitting MEM_WRITE(0xCAFEBABE) with SYNC...\n");
    {
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
                                            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 42 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
        if (r) { printf("❌ MEM_WRITE caused GPU fault!\n"); goto end; }
    }

    /* Invalidate cache before reading result */
    volatile uint32_t *res = (volatile uint32_t *)(buf + 0x1000/4);
    cache_invalidate((void*)res, 64);
    
    printf("res[0]=0x%08x (before cache clean)\n", res[0]);
    
    /* Also try reading multiple times */
    for (int i = 0; i < 5; i++) {
        cache_invalidate((void*)res, 64);
        printf("  [%d] res[0]=0x%08x\n", i, res[0]);
        if (res[0] == 0xCAFEBABE) {
            printf("✅✅✅ GPU MEM_WRITE CONFIRMED!\n");
            goto end;
        }
        if (res[0] != 0xABABABAB) {
            printf("⚠️  res[0] changed to unexpected value: 0x%08x\n", res[0]);
            goto end;
        }
    }

    /* Try mapping with MAP_SHARED | MAP_ANONYMOUS */
    printf("\n[*] Retrying with MAP_SHARED...\n");
    {
        uint32_t *buf2 = mmap(NULL, 0x2000, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (buf2 == MAP_FAILED) { printf("[-] mmap shared: %s\n", strerror(errno)); goto end; }
        memset(buf2, 0x42, 0x2000);
        
        uint64_t ga2 = 0;
        struct kgsl_map_user_mem m2 = { .len = 0x2000, .hostptr = (unsigned long)buf2,
                                         .memtype = KGSL_USER_MEM_TYPE_ADDR };
        if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m2)) {
            printf("[-] map2: %s\n", strerror(errno)); goto end;
        }
        ga2 = m2.gpuaddr;
        printf("[+] buf2: cpu=%p gpu=0x%lx\n", buf2, (unsigned long)ga2);
        
        /* Build commands */
        uint32_t *p2 = buf2;
        *p2++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        *p2++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
        *p2++ = lower_32_bits(ga2 + 0x1000);
        *p2++ = upper_32_bits(ga2 + 0x1000);
        *p2++ = 0xCAFEBABE;
        *p2++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
        *p2++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd2_bytes = (char*)p2 - (char*)buf2;
        
        __builtin___clear_cache((char*)buf2, (char*)buf2 + 0x2000);
        
        /* Flush+invalidate before submission */
        cache_invalidate((void*)buf2, 0x2000);
        
        struct kgsl_command_object obj2 = { .gpuaddr = ga2, .size = cmd2_bytes,
                                             .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req2 = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj2, .cmdsize = sizeof(obj2), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 43 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req2);
        printf("submit2: %s ts=%u\n", r ? strerror(errno) : "OK", req2.timestamp);
        if (r) goto end;
        
        cache_invalidate((void*)(buf2 + 0x1000/4), 64);
        printf("  buf2[0x1000/4]=0x%08x\n", buf2[0x1000/4]);
        if (buf2[0x1000/4] == 0xCAFEBABE) {
            printf("✅✅✅ MAP_SHARED: GPU MEM_WRITE CONFIRMED!\n");
        }
    }

end:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    printf("[+] DONE\n");
    return 0;
}