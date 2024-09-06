import setuptools

scripts = ["./bin/get_fg_bias"]

setuptools.setup(
    name="fgbias",
    version="0.0.1",
    author="Niall MacCrann",
    author_email="nm746@cam.ac.uk",
    description="Code for calculating CMB lensing foreground biases",
    packages=["fgbias"],
    include_package_data=True,
    package_data={'fgbias': ['defaults.yaml']},
    scripts=scripts
)