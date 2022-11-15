# This file is part of scarlet_lite.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


class Cache:
    """Cache to hold all complex proximal operators, transformation etc.

    Convention to use is that the lookup `name` refers to the class or method
    that pushes content onto the cache, the `key` can be chosen at will.

    """

    _cache = {}

    @staticmethod
    def check(name, key):
        try:
            Cache._cache[name]
        except KeyError:
            Cache._cache[name] = {}
        return Cache._cache[name][key]

    @staticmethod
    def set(name, key, content):
        try:
            Cache._cache[name]
        except KeyError:
            Cache._cache[name] = {}
        Cache._cache[name][key] = content

    def __repr__(self):
        repr(Cache._cache)
