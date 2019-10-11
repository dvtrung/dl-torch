make html
cp -a build/html/* build
rm -r build/html
rm -r build/doctrees
cd build

if [[ -f ".git" ]]; then
  touch .nojekyll
  git checkout gh-pages
  git add .
  git commit -m "update docs"
  git push origin HEAD
fi