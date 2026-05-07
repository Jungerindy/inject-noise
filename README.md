# Information Hiding 2026: Project 3
## Injecting acquisition noise in the raw domain
Natural steganography uses a noise model for the raw domain.
Researchers often simplify by working in the developed domain.
### Task
- Acquire the raw domain and denoise it.
- Re-inject an acquisition noise of a known profile.
- Benchmark w. r. t. noise variance.

### Pecularities
- Prepare as a patch for dcraw. Alternatively, use rawpy Python package.
- Use ALASKA raws

Based on the work of P. Bas. Steganography via cover-source switching. IEEE WIFS, 2016.

### Useful commands

insert noise and export result as .tif
-Y dennotes the usage of my stego patch
a_base b_base a_target b_target
```bash
./build/dcraw -Y 9.88e-08 2.61e-09 0.00059563 6.8e-07 -v -T -4 -W -q 3 sample1.dng 
```
export .dng as .ppm in dcraw
```bash
./dcraw -w sample1.dng 
```

export .dng as .tif
```bash
./dcraw -T -4 -W -q 3 sample1.dng 
```

