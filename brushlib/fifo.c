/*
 *
 * Based on fifo.c from syslog-win32 by Alexander Yaworsky
 * http://syslog-win32.svn.sourceforge.net/viewvc/syslog-win32/tags/syslog-win32-0-3/daemon/fifo.c
 *
 * THIS SOFTWARE IS NOT COPYRIGHTED
 *
 * This source code is offered for use in the public domain. You may
 * use, modify or distribute it freely.
 *
 * This code is distributed in the hope that it will be useful but
 * WITHOUT ANY WARRANTY. ALL WARRANTIES, EXPRESS OR IMPLIED ARE HEREBY
 * DISCLAIMED. This includes but is not limited to warranties of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#include <stdlib.h>
#include "fifo.h"

struct fifo_item
{
    struct fifo_item *next;  /* queue is a single-linked list */
    void *payload;
};

struct fifo
{
    struct fifo_item *first;  /* first pushed item */
    struct fifo_item *last;   /* last pushed item */
    int item_count;
};

/*
 * fifo_new:
 *
 * Allocate and initialize fifo structure. Add an empty item to the fifo.
 */
struct fifo* fifo_new()
{
    struct fifo *ret = (struct fifo *) malloc(sizeof(struct fifo));
    ret->first = NULL;
    ret->last = NULL;
    ret->item_count = 0;
    return ret;
}

/*
 * fifo_free:
 *
 * Delete all items and free fifo structure.
 */
void fifo_free(struct fifo* queue, FifoUserFreeFunction user_free)
{
    struct fifo_item *item;

    while ((item = queue->first) != NULL)
    {
        queue->first = item->next;
        user_free(item);
    }
    free(queue);
}

/*
 * fifo_push:
 *
 * Add item to queue.
 */
void fifo_push(struct fifo* queue, void* data)
{
    struct fifo_item *item = (struct fifo_item*) malloc(sizeof(struct fifo_item));
    item->next = NULL;
    item->payload = data;
    if (!queue->last)
        queue->first = item;
    else
        queue->last->next = item;
    queue->last = item;
}

/*
 * fifo_pop:
 *
 * Extract item from queue. Returns NULL on empty queue.
 */
void* fifo_pop(struct fifo* queue)
{
    struct fifo_item *item;
    void *data;

    item = queue->first;
    if (!item)
        return NULL;

    queue->first = item->next;
    if (!queue->first)
        queue->last = NULL;

    data = item->payload;
    free(item);
    queue->item_count--;
    return data;
}

void* fifo_peek_first(struct fifo *queue) {
    return (!queue->first) ? NULL : queue->first->payload;
}

void* fifo_peek_last(struct fifo *queue) {
    return (!queue->last) ? NULL : queue->last->payload;
}
