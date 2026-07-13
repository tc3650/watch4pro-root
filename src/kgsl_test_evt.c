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
#define CP_EVENT_WRITE 0x1B
#define CP_MEM_WRITE 0x3d

#define lo32(x) ((uint32_t)((x)&0xFFFFFFFF))
#define hi32(x) ((uint32_t)(((x)>>16)>>16))

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open\n"); return 1; }
    printf("[+] fd=%d\n", fd);

    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) { printf("[-] ctx\n"); close(fd); return 1; }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    uint32_t *buf = mmap(NULL, 0x2000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(buf, 0x77, 0x2000);

    uint64_t base_ga = 0;
    {	struct kgsl_map_user_mem m = { .len = 0x2000, .hostptr = (unsigned long)buf,
        	.memtype = KGSL_USER_MEM_TYPE_ADDR };
    	if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) { printf("[-] map\n"); goto end; }
    	base_ga = m.gpuaddr; }
    printf("[+] cpu=%p gpu=0x%lx\n", buf, (unsigned long)base_ga);

    uint64_t rs_ga = base_ga + 0x1000;
    buf[0x1000/4] = 0x11111111;

    /* Try 6 different event write formats */
    printf("\n== Testing CP_EVENT_WRITE (6 formats) ==\n");
    uint32_t events[] = {0x00, 0x01, 0x02, 0x04, 0x10, 0x20, 0x100, 0x1E, 0x1F};
    for (int ei = 0; ei < 9; ei++) {
        buf[0x1000/4] = 0x11111111;
        
        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        /* CP_EVENT_WRITE with 4 dwords: event, addr_lo, addr_hi, value */
        *p++ = CP_TYPE7_PKT(CP_EVENT_WRITE, 4);
        *p++ = events[ei];
        *p++ = lo32(rs_ga + ei*4);
        *p++ = hi32(rs_ga + ei*4);
        *p++ = 0xDEADBEEF + ei;
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;
        
        __builtin___clear_cache((char*)buf, (char*)buf + 0x2000);
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 10+ei };
        int r = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("  event=0x%02x: %s ts=%u rs[%d]=0x%08x\n", events[ei],
               r ? strerror(errno) : "OK", req.timestamp, ei, buf[0x1000/4 + ei]);
        if (buf[0x1000/4] == 0xDEADBEEF || buf[0x1004/4] == 0xDEADBEEF + 1)
            { printf("✅ EVENT_WRITE WORKS with event=0x%02x!\n", events[ei]); break; }
    }

    /* Try MEM_WRITE with count=2 (no value, just addr) - maybe it reads the value from reg */
    printf("\n== Testing MEM_WRITE alternative formats ==\n");
    {
        buf[0x1000/4] = 0x22222222;
        uint32_t *p = buf;
        *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
        /* MEM_WRITE with count=2: addr_lo, addr_hi - writes from implicit source */
        *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 2);
        *p++ = lo32(rs_ga);
        *p++ = hi32(rs_ga);
        *p++ = CP_TYPE7_PKT(CP_NOP, 0);
        int cmd_bytes = (char*)p - (char*)buf;
        
        struct kgsl_command_object obj = { .gpuaddr = base_ga, .size = cmd_bytes,
            .flags = KGSL_CMDLIST_IB };
        struct kgsl_gpu_command req = { .flags = KGSL_COMMAND_SYNC,
            .cmdlist = (uint64_t)(unsigned long)&obj, .cmdsize = sizeof(obj), .numcmds = 1,
            .context_id = ctx.drawctxt_id, .timestamp = 99 };
        ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        printf("  MEM_WRITE(count=2): rs[0]=0x%08x\n", buf[0x1000/4]);
        if (buf[0x1000/4] != 0x22222222) printf("  changed!\n");
    }

end:
    { struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
      ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d); }
    close(fd);
    return 0;
}