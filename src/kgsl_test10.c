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
#define CP_REG_TO_MEM 0x60
#define CP_MEM_WRITE 0x3d

#define lo32(x) ((uint32_t)((x)&0xFFFFFFFF))
#define hi32(x) ((uint32_t)(((x)>>16)>>16))

static inline void cache_sync(void *addr, int len) {
    void *a = (void*)((uintptr_t)addr & ~0xFFF);
    size_t l = ((uintptr_t)addr + len + 0xFFF) & ~0xFFF;
    l -= (uintptr_t)a;
    mprotect(a, l, PROT_NONE);
    mprotect(a, l, PROT_READ|PROT_WRITE);
}

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open\n"); return 1; }
    printf("[+] fd=%d\n", fd);

    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) { printf("[-] ctx\n"); close(fd); return 1; }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* Allocate buffer: cmds at 0, result at 0x1000 */
    uint32_t *buf = mmap(NULL, 0x2000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(buf, 0xBB, 0x2000);

    /* Map to GPU */
    uint64_t base_ga = 0;
    {	struct kgsl_map_user_mem m = { .len = 0x2000, .hostptr = (unsigned long)buf,
        	.memtype = KGSL_USER_MEM_TYPE_ADDR };
    	if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) { printf("[-] map\n"); goto end; }
    	base_ga = m.gpuaddr; }
    printf("[+] mapped: cpu=%p gpu=0x%lx\n", buf, (unsigned long)base_ga);

    uint64_t dst_ga = base_ga + 0x1000;

    /* === Test 1: CP_REG_TO_MEM - read GPU register 0x0000 (GPU_ID) into memory === */
    printf("\n== Test 1: CP_REG_TO_MEM (GPU_ID reg) ==\n");
    {
        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        /* CP_REG_TO_MEM: count=2 + 1 = 3? Let's try different formats */
        /* Format A: header(1) + reg(1) + mem_lo(1) + mem_hi(1) = count=3 */
        *p++ = CP_TYPE7_PKT(CP_REG_TO_MEM, 3);
        *p++ = 0x00000000; /* GPU_ID register (0 = CP register bank) */
        *p++ = lo32(dst_ga);
        *p++ = hi32(dst_ga);
        *p++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;
        
        __builtin___clear_cache((char*)buf, (char*)buf + 0x2000);
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 1 };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);
        
        cache_sync((void*)(buf + 0x1000/4), 8);
        printf("  dst[0]=0x%08x (GPU_ID?)\n", buf[0x1000/4]);
        if (buf[0x1000/4] != 0xBBBBBBBB)
            printf("✅ GPU REG TO MEM WORKS! GPU_ID=0x%08x\n", buf[0x1000/4]);
    }
    
    /* === Test 2: Try different REG_TO_MEM format === */
    printf("\n== Test 2: CP_REG_TO_MEM (count=2, no mem_hi) ==\n");
    {
        buf[0x1000/4] = 0xCCCCCCCC;
        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        /* Format B: header(1) + reg(1) + mem(1) = count=2 (32-bit GPU addr?) */
        *p++ = CP_TYPE7_PKT(CP_REG_TO_MEM, 2);
        *p++ = 0x00000000; /* GPU_ID */
        *p++ = lo32(dst_ga); /* mem (assumes 32-bit GPU addr) */
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 2 };
        ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        
        cache_sync((void*)(buf + 0x1000/4), 8);
        printf("  dst[0]=0x%08x\n", buf[0x1000/4]);
        if (buf[0x1000/4] != 0xCCCCCCCC)
            printf("✅ Format B works! GPU_ID=0x%08x\n", buf[0x1000/4]);
    }
    
    /* === Test 3: Try CP_MEM_WRITE with write to different offset within same page === */
    printf("\n== Test 3: CP_MEM_WRITE with CP_WAIT_FOR_ME prefix ==\n");
    {
        buf[0x1000/4] = 0xDDDDDDDD;
        uint32_t *p = buf;
        /* Add WAIT_FOR_ME before the write (from cheese code) */
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        /* WAIT_FOR_ME */
        *p++ = 0x70000000 | (0x13 << 16); /* CP_WAIT_FOR_ME with count=0 */
        /* MEM_WRITE */
        *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
        *p++ = lo32(dst_ga + 4);
        *p++ = hi32(dst_ga + 4);
        *p++ = 0xDEADBEEF;
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;
        
        __builtin___clear_cache((char*)buf, (char*)buf + 0x2000);
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 3 };
        ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        
        cache_sync((void*)(buf + 0x1000/4), 8);
        printf("  dst[0]=0x%08x dst[1]=0x%08x\n", buf[0x1000/4], buf[0x1004/4]);
        if (buf[0x1004/4] == 0xDEADBEEF)
            printf("✅ MEM_WRITE WORKS with WAIT_FOR_ME!\n");
    }
    
    /* === Test 4: Try CP_NOP with GFX decode === */
    printf("\n== Test 4: CP_NOP with various base values ==\n");
    /* Some GPUs use CP_NOP with a base address that modifies behavior */
    {
        buf[0x1000/4] = 0xEEEEEEEE;
        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_NOP, 1);
        *p++ = 0x0; /* zero */
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
        *p++ = lo32(dst_ga);
        *p++ = hi32(dst_ga);
        *p++ = 0xCAFEBABE;
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 4 };
        ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        
        cache_sync((void*)(buf + 0x1000/4), 8);
        printf("  dst[0]=0x%08x\n", buf[0x1000/4]);
        if (buf[0x1000/4] == 0xCAFEBABE)
            printf("✅ MEM_WRITE works!\n");
    }

end:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    return 0;
}