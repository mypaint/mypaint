#!/bin/sh
# This script is written to be run on Travis CI,
# triggering the appimage as the final step of
# building the master branch of the MyPaint repo.

# Make sure we don't leak the token
set +x

if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$TRAVIS_BRANCH" != "master" ]; then
    echo " Skipping appimage build for pull requests and non-master builds"
    exit 0
fi

if [ -z "$TRAVIS_API_TOKEN" ]; then
    echo "Travis api token not set, cannot trigger appimage build!"
    exit 0
fi

echo "== Trigger appimage build =="
body='{
"request": {
"branch":"master"
}}'

curl -s -X POST \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Travis-API-Version: 3" \
     -H "Authorization: token $TRAVIS_API_TOKEN" \
     -d "$body" \
     https://api.travis-ci.org/repo/mypaint%2Fmypaint-appimage/requests

exit 0
