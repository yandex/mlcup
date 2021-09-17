set -e

echo "Downloading pip packages for offline installation in Yandex.Contest"
echo "You might condider deleting the downloaded versions of common packages (such as numpy, pandas, pytorch, transformers, etc.) to reduce the solution size as they are already installed in the testing environment"

# download the needed python modules for installation in the checker
pip download -r requirements.txt --platform manylinux1_x86_64  --python-version 3.8 --only-binary=:all: -d packages 

echo "Compressing the solution into a tar"
rm baseline_solution.tar.gz || true  # delete the baseline solution if it exists
tar -czvf baseline_solution.tar.gz *
