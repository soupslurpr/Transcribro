{ pkgs ? import <nixpkgs> {} }:
 
let fhs = pkgs.buildFHSUserEnv {
  name = "android-env";
  targetPkgs = pkgs: with pkgs;
    [ git
      gitRepo
      gnupg
      python2
      curl
      procps
      openssl
      gnumake
      nettools
      # For nixos < 19.03, use `androidenv.platformTools`
      androidenv.androidPkgs_9_0.platform-tools
      jdk
      schedtool
      util-linux
      m4
      gperf
      perl
      libxml2
      zip
      unzip
      bison
      flex
      lzop
      python3
      android-studio
    ];
  multiPkgs = pkgs: with pkgs;
    [ zlib
      ncurses5
    ];
  runScript = "bash";
  profile = ''
    export ALLOW_NINJA_ENV=true
    export USE_CACHE=1
    export ANDROID_JAVA_HOME=${pkgs.jdk.home}sdkmanager install avd
    export LD_LIBRARY_PATH=/usr/lib:/usr/lib32
  '';
};
in pkgs.stdenv.mkDerivation {
  name = "android-env-shell";
  nativeBuildInputs = [ fhs ];
  shellHook = "exec android-env";

}