/*
    Copyright (C) 2019  Grzegorz Wójcik

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include <iostream>

namespace terminal {
const char *const reset = "\033[1;0m";
const char *const black = "\033[1;30m";
const char *const red = "\033[1;31m";
const char *const green = "\033[1;32m";
const char *const yellow = "\033[1;33m";
const char *const blue = "\033[1;34m";
const char *const magenta = "\033[1;35m";
const char *const cyan = "\033[1;36m";
const char *const white = "\033[1;37m";
}
template <int out = 1, typename... T> inline void Log(T... args) {
    if (out == 1) {
        int dummy[sizeof...(T)] = {(std::cout << args,1)...};
        (void)(dummy);
    } else {
        int dummy[sizeof...(T)] = {(std::cerr << args,1)...};
        (void)(dummy);
    }
};

template <typename... T> inline void LogErr(T... args) {
    Log<2>(terminal::red, "[ERR] ", args..., terminal::reset, '\n');
};
template <typename... T> inline void LogDev(T... args) {
    Log<1>(terminal::yellow, "[DEV] ", args..., terminal::reset, '\n');
};
template <typename... T> inline void LogInfo(T... args) {
    Log<1>(terminal::green, "[INF] ", args..., terminal::reset, '\n');
};
