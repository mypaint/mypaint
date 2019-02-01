Interactive RGB Transparency: a color rendering tool for superimposed translucent layers in digital images
* https://www.researchgate.net/publication/322927901_Interactive_RGB_Transparency_a_color_rendering_tool_for_superimposed_translucent_layers_in_digital_images

Subtractive color mixing
* http://scottburns.us/subtractive-color-mixture/
* http://www.handprint.com/HP/WCL/color3.html#mixprofile

The weighted geometric mean (or the weighted power mean) is a generally recognized method to predict the appearance of mixed paints.
However, this only really works well with a wide multi-channel spectrum because Grassman's law no longer applies:
* https://en.wikipedia.org/wiki/Grassmann%27s_laws_(color_science) 

Additive RGB works in a linear fashion and similary to the real world, wheras Subtractive RGB is non-linear and inverts the model.
The emissions become reflectance, and where (1,0,0) may be a perfectly valid real-world red light, it is certainly not a valid
real-world reflectance for a natural red object.

Spectral from RGB

* http://scottburns.us/reflectance-curves-from-srgb/
* https://jo.dreggn.org/home/2015_spectrum.pdf
* https://jo.dreggn.org/home/2018_manuka.pdf
* https://github.com/colour-science/smits1999

Mypaint uses a modified Meng function to be more similar (identical?) to Scott Allen Burns method.  By providing a D65 illuminant
it seems to be simpler.  See the accompanying python file

Mypaint could be modified to generate these tables on load and for a variety of colorspaces, however sRGB Rec 709
is currently the most feasible unless the technique is changed dramatically.  This is because color spaces such as
Rec 2020 use spectral primaries that cannot be represented by a smooth spectral emission or reflectance unless you
account for brightness.

We probably should solve for energy conservation as shown here, but I'm not sure if it is really necessary for our purposes. 
* http://scottburns.us/fast-rgb-to-spectrum-conversion-for-reflectances/

So the general procedure, once you have the  r, g, b, and spectral->RGB matrix prepared, is sort of simple:

* un-premultiply alpha if necessary (Is this absolutely necessary?)
* multiply each RGB component with the corresponding spectral curve
* sum the three curves along the wavelength axis to obtain one reflectance curve for the color
* repeat for the other color you plan to blend with
* calculate the ratio based on the original alpha components, normalize to sum to 1.0
 * if color_1 was alpha 25% and color_2 was alpha 50%, the resulting ratio would be 0.333 (since it is 1:2 ratio)
* perform the weighted geometric mean for the 2 colors: color_1[i]**(ratio) * color_2[i]**(1-ratio)
* multiply the resultant curve against the Spectral_to_RGB matrix to obtain RGB
* calculate a new alpha channel in the normal fashion
* multiply the new alpha against the RGB to store premultiplied color
* done
