# Scalar wave generation from a neutron star binary
This code provides a PDE solver for the equation of motion of a self-interacting
 scalar field with a curvature coupling, where the spacetime curvature is that
 of a neutron star binary. The treatment is not fully relativistic, but
 implemented as a simple modification to the potential term by an exponential
 ansatz. See *derivation* for the precise definitions.
![](./gif/animated.gif)

## Usage
The script **Wave_3D_star.py** generates a set of four-dimensional grids
and **NS_plot_3D.py** converts the grids to images. Gifs are easily produced for
example with ImageMagick:

```console
foo@bar:~$  convert -delay 12 *jpg animated.gif
```
Cropping the images can also be done on the command line with e.g.
```console
foo@bar:~$  mogrify -crop 640x240+20+120 *.jpg
```
