The files in this folder are used to generate nightly Flatpak builds of MyPaint.

Stable Flatpak builds of MyPaint are on [flathub.org](https://raw.githubusercontent.com/mypaint/mypaint/master/flatpak/mypaint-stable.flatpakref).

To locally build and install nightly MyPaint using Faltpak:

    flatpak remote-add --no-gpg-verify --user local-repo ${PATH_TO_REPO}
    flatpak-builder --repo=${PATH_TO_REPO} ${PATH_TO_BUILD} org.mypaint.MyPaint.json
    flatpak --user install local-repo org.mypaint.MyPaint-Nightly

Set PATH_TO_REPO and PATH_TO_BUILD to somewhere in your home.  The
directories will be created by flatpak if they don't exist.
