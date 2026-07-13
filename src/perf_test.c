#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

int main() {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 0;
    pe.exclude_hv = 0;
    
    int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd < 0) { printf("perf_event_open: %s\n", strerror(errno)); return 1; }
    printf("perf_event_open OK fd=%d\n", fd);
    
    size_t mmap_size = (1 + 16) * sysconf(_SC_PAGESIZE);
    void *buf = mmap(NULL, mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (buf == MAP_FAILED) { printf("mmap: %s\n", strerror(errno)); close(fd); return 1; }
    printf("mmap OK %p\n", buf);
    
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    usleep(500000);
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    
    long long count;
    read(fd, &count, sizeof(count));
    printf("cycles: %lld\n", count);
    
    munmap(buf, mmap_size);
    close(fd);
    return 0;
}
