@ECHO off
SET MYPAINT_DEBUG=1
CD %~dp0\..
bin\python2.exe bin\mypaint %*
PAUSE
