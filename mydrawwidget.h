#ifndef __MYDRAWWIDGET_H__
#define __MYDRAWWIDGET_H__

#include <gtk/gtk.h>
#include "surface.h"
#include "brush.h"

#define MYDRAWWIDGET(obj)          GTK_CHECK_CAST (obj, mydrawwidget_get_type (), MyDrawWidget)
#define MYDRAWWIDGET_CLASS(klass)  GTK_CHECK_CLASS_CAST (klass, mydrawwidget_get_type (), MyDrawWidgetClass)
#define IS_MYDRAWWIDGET(obj)       GTK_CHECK_TYPE (obj, mydrawwidget_get_type ())

typedef struct _MyDrawWidget       MyDrawWidget;
typedef struct _MyDrawWidgetClass  MyDrawWidgetClass;

struct _MyDrawWidget
{
  GtkDrawingArea da; // was: vbox

  Surface * surface;
  Brush * brush;
};

struct _MyDrawWidgetClass
{
  GtkDrawingAreaClass parent_class;

  //void (* mydrawwidget) (MyDrawWidget *mdw);
};

GType mydrawwidget_get_type (void);
GtkWidget* mydrawwidget_new (void);

void mydrawwidget_clear (MyDrawWidget *mdw);
// brush should not be freed before a new one is set
void mydrawwidget_set_brush (MyDrawWidget *mdw, Brush * brush);

/*
void mydrawwidget_set_content (MyDrawWidget *mdw, Bitmap * src);
void mydrawwidget_get_content (MyDrawWidget *mdw, Bitmap * dst);
*/

#endif /* __MYDRAWWIDGET_H__ */
