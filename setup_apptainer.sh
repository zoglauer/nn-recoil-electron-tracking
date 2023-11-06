cp -pr ~/.local /global/scratch/users/$USER/.local
rm -rf ~/.local
ln -s /global/scratch/users/$USER/.local ~/.local

# build pytorch container
apptainer build container.sif savio/pytorch.def