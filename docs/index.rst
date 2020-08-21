.. toctree::
   :hidden:
   :glob:
   :titlesonly:

   source/compiling.rst
   source/generating.rst


ROSCO toolbox documentation
===========================
.. only:: html

   :Version: |release|
   :Date: |today|

NREL's Reference OpenSource Controller (ROSCO) toolbox for wind turbine applications is a toolbox designed to ease controller implementation for the wind turbine researcher. The purpose of these documents is to provide information for the use of the ROSCO related toolchain. 

Standard Use
-------------
For the standard use case in OpenFAST, ROSCO will need to be compiled. This is made possible via the instructions found in :ref:`compiling`. Once the controller is compiled, the turbine model needs to point to the compiled binary. In OpenFAST, this is ensured by changing the :code:`DLL_FileName` parameter in the ServoDyn input file. 

Additionally, an additional input file is needed for the ROSCO controller. Though the controller only needs to be compiled once, each individual turbine/controller tuning requires an input file. This input file is generically dubbed "DISCON.IN''. In OpenFAST, the :code:`DLL_InFile` parameter should be set to point to the desired input file. The ROSCO toolbox is used to automatically generate the input file. These instructions are provided in the instructions for :ref:`generate_discon`.


License
-------
Copyright 2020 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.