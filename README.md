[TODO: ZENODO BADGE]

# EntropyGrad.jl
This Julia package is companion code which contains the examples described in [TODO: PAPER LINK]

In its current state, this package serves two purposes:
- Complimentary role to the paper, providing code samples to those curious about the actual implementation.
- Reproduce figures found in the paper by providing a documented (via comments) step-by-step walkthrough of the numerical experiments in the paper.

While the package is not meant for nor setup for user-friendly general use, the interested reader should nevertheless find the code to be straightforward to generalize since it is mostly based on implementing the contour integral formulation correctly, followed by use of automatic differentiation.

![inter_gif](https://github.com/user-attachments/assets/a02fb55b-b8c1-4368-867f-dabe1a7ef0ce){ width=300px }


# Installation

The current version has been tested with Julia v1.10.0. Newer Julia releases may work if all the appropriate dependencies can be resolved.

As this package relies on packages registered in the ACEsuit family which may not yet have been registered in General, you will first have to follow the instructions to add the ACE registry to Julia: https://github.com/ACEsuit/ACEregistry

As an unregistered Julia package, you can install this package via ```] add https://github.com/tinatorabi/EntropyGrad.jl``` or alternatively ```Pkg.add(PackageSpec(url="https://github.com/tinatorabi/EntropyGrad.jl"))```. Then explore the files in the examples folder.

# Using the package

The examples folder contains scripts with detailed comments. Evaluating line-by-line from the top walks the user through the process of obtaining the solutions described in the paper. Along with the comments in the src folder this should be sufficient to generalize it to other use cases. Keep in mind that to run the examples, you will have to install the packages specified via ```using``` at the top of each example which are also part of the dependencies of this package.

# References

[TODO: ADD PAPER LINK]
