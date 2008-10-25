// FIXME: maybe this file should not be part of mypaintlib?

static const int colorselector_size = 240; // diameter of Swiss Cheese Wheel Color Selector(TM)
static const int colorselector_center = (colorselector_size/2); // radii/center coordinate of SCWCS

#include <cmath> // atan2, sqrt or hypot

/*
  --------- Swiss Cheese Wheel Color Selector(TM) --------- 

  Ring 0: Current brush color
  Ring 1: Value
  Ring 2: Saturation
  Ring 3: Hue
  
*/

// Frequently used constants
static const float RAD_TO_ONE = 0.5f/M_PI;
static const float TWO_PI = 2.0f*M_PI;
// Calculate these as precise as the hosting system can once and for all
static const float ONE_OVER_THREE = 1.0f/3.0f;
static const float TWO_OVER_THREE = 2.0f/3.0f;

// 1 Mile of variables....
void get_scwcs_hsva_at( float* h, float* s, float* v, float* a, float x, float y, float base_h, float base_s, float base_v, bool adjust_color = true, bool only_colors = true, float mark_h = 0.0f )
{
  float rel_x = (colorselector_center-x);
  float rel_y = (colorselector_center-y);
  
  //float radi = sqrt( rel_x*rel_x + rel_y*rel_y ); // Pre-C99 solution
  float radi = hypot( rel_x, rel_y );
  float theta = atan2( rel_y, rel_x );
  if( theta < 0.0f ) theta += TWO_PI; // Range: [ 0, 2*PI )
  
  // Current brush color
  *h = base_h;
  *s = base_s;
  *v = base_v;
  *a = 255.0f; // Alpha is always [0,255]
  
  if( radi < 43.0f || radi > 120.0f ) // Masked/Clipped/Tranparent area
  {
    // transparent/cut away
    *a = 0.0f;
  }
  else if( radi > 50.0f && radi <= 65.0f ) // Saturation
  {
    *s = (theta/TWO_PI);
    
    if( only_colors == false && floor(*s*255.0f) == floor(base_s*255.0f) ) {
      // Draw marker
      *s = *v = 1.0f;
      *h = mark_h;
    }
    
  }
  else if( radi > 65.0f && radi <= 90.0f ) // Value 
  {
    *v = (theta/TWO_PI);
    
    if( only_colors == false && floor(*v*255.0f) == floor(base_v*255.0f) ) {
      // Draw marker
      *s = *v = 1.0f;
      *h = mark_h;
    }
    
  }
  else if( radi > 90.0f && radi <= 120.0f ) // Hue
  {
    *h = (theta*RAD_TO_ONE);
    
    if( only_colors == false && floor(*h*360.0f) == floor(base_h*360.0f) ) {
      // Draw marker
      *h = mark_h;
    }
    
    if( adjust_color == false ) {
      // Picking a new hue resets Saturation and Value
      *s = *v = 1.0f;
    }
  }
}

PyObject* pick_scwcs_hsv_at( float x, float y, double brush_h, double brush_s, double brush_v )
{
  float h,s,v,a;
  
  float base_h = brush_h;
  float base_s = brush_s;
  float base_v = brush_v;
  
  get_scwcs_hsva_at(&h, &s, &v, &a, x, y, base_h, base_s, base_v);
  
  return Py_BuildValue("fff",h,s,v);
}

void render_swisscheesewheelcolorselector(PyObject * arr, double brush_h, double brush_s, double brush_v)
{
  assert(PyArray_ISCARRAY(arr));
  assert(PyArray_ISBEHAVED(arr));
  assert(PyArray_NDIM(arr) == 3);
  assert(PyArray_DIM(arr, 0) == colorselector_size);
  assert(PyArray_DIM(arr, 1) == colorselector_size);
  assert(PyArray_DIM(arr, 2) == 4);  // memory width of pixel data ( 3 = RGB, 4 = RGBA )
  guchar* pixels = (guchar*)((PyArrayObject*)arr)->data;
  
  const int pixels_inc = PyArray_DIM(arr, 2);
  
  float h,s,v,a;
  
  float base_h = brush_h;
  float base_s = brush_s;
  float base_v = brush_v;
  
  float ofs_h = ((base_h+ONE_OVER_THREE)>1.0f)?(base_h-TWO_OVER_THREE):(base_h+ONE_OVER_THREE); // offset hue

  for(float y=0; y<colorselector_size; y++) {
    for(float x=0; x<colorselector_size; x++) {
      get_scwcs_hsva_at(&h, &s, &v, &a, x, y, base_h, base_s, base_v, false, false, ofs_h);
      hsv_to_rgb_range_one(&h,&s,&v); // convert from HSV [0,1] to RGB [0,255]
      pixels[0] = h; pixels[1] = s; pixels[2] = v; pixels[3] = a;
      pixels += pixels_inc; // next pixel block
    }
  }
}
