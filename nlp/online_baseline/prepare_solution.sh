set -e

echo "Downloading pip packages for offline installation in Yandex.Contest"

# note: this assert is not a strict regulation, but a kind notice that would help you reduce the size of your solution
python3 -c "overlapped_requirements = set(open('preinstalled_packages.txt')) & set(open('requirements.txt')); assert len(overlapped_requirements) == 0, 'Please, don\'t try installing packages that have been pre-installed in Yandex.Contest, found: {}'.format(overlapped_requirements)"

# download the needed python modules for installation in the checker
pip download -r requirements.txt --platform manylinux1_x86_64  --python-version 3.8 --only-binary=:all: -d packages --no-deps 

echo "Compressing the solution into a tar"
rm baseline_solution.tar.gz || true  # delete the baseline solution if it exists
tar -czvf baseline_solution.tar.gz *
