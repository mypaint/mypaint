#ifndef FIFO_H
#define FIFO_H

typedef struct fifo Fifo;
typedef void (*FifoUserFreeFunction) (void *item_data);

Fifo* fifo_new();
void fifo_free(Fifo* self, FifoUserFreeFunction data_free);

void fifo_push(Fifo* self, void* data);
void* fifo_pop(Fifo* self);

#endif // FIFO_H
