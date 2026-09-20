# SDRplay API headers, version 3.15

The C headers under `inc/` are SDRplay's, copied unmodified from a Windows install of
the SDRplay Hardware API 3.15, at `C:\Program Files\SDRplay\API\inc\`.

They are here so that `tools/generate_sdrplay_api.py` can regenerate
`lib/buzz/receiver/sdrplay_api.py` from the same source on any machine, including one
with no receiver and no API installed.  Without them the generated bindings could only be
checked where the API happens to be present, which is neither CI nor most contributors.

**These headers are not the library.** An operator still installs the SDRplay Hardware
API themselves, from <https://sdrplay.com/hardware-api/>, under "Other" on that site.
That is what supplies `sdrplay_api.dll` on Windows or `libsdrplay_api.so` on Linux, and
the `SDRplayAPIService` background service the API talks through.  Having SDRconnect
working proves nothing about this: SDRconnect ships its own USB drivers and speaks to
the hardware directly, and it installs neither the API nor the service.

Nothing in this project modifies the headers.  To move to a later API, install it, copy
its `inc/` over a new directory beside this one, and regenerate.

## License

The headers themselves carry no per-file banner.  What follows is the "Legal
Information" section of `SDRplay_API_Specification_v3.15.pdf`, shipped in the same
install under `docs/`, reproduced here so that the terms travel with the code as
condition 1 requires.  The copyright holder is named in the last two paragraphs:
SDRplay Limited, a company registered in England, number 09035244.

The first three conditions and the disclaimer are the BSD 3-Clause license.

```
Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list
of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be
used to endorse or promote products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SDRPlay modules use a Mirics chipset and software. The information supplied hereunder
is provided to you by SDRPlay under license from Mirics. Mirics hereby grants you a
perpetual, worldwide, royalty free license to use the information herein for the
purpose of designing software that utilizes SDRPlay modules, under the following
conditions:

There are no express or implied copyright licenses granted hereunder to design or
fabricate any integrated circuits or integrated circuits based on the information in
this document. Mirics reserves the right to make changes without further notice to any
of its products. Mirics makes no warranty, representation or guarantee regarding the
suitability of its products for any particular purpose, nor does Mirics assume any
liability arising out of the application or use of any product or circuit, and
specifically disclaims any and all liability, including without limitation
consequential or incidental damages. Typical parameters that may be provided in Mirics
data sheets and/or specifications can and do vary in different applications and actual
performance may vary over time. All operating parameters must be validated for each
customer application by the buyer's technical experts. SDRPlay and Mirics products are
not designed, intended, or authorized for use as components in systems intended for
surgical implant into the body, or other applications intended to support or sustain
life, or for any other application in which the failure of the Mirics product could
create a situation where personal injury or death may occur. Should Buyer purchase or
use SDRPlay or Mirics products for any such unintended or unauthorized application,
Buyer shall indemnify and hold both SDRPlay and Mirics and their officers, employees,
subsidiaries, affiliates, and distributors harmless against all claims, costs, damages,
and expenses, and reasonable attorney fees arising out of, directly or indirectly, any
claim of personal injury or death associated with such unintended or unauthorized use,
even if such claim alleges that either SDRPlay or Mirics were negligent regarding the
design or manufacture of the part. Mirics FlexiRF, Mirics FlexiTV and Mirics are
trademarks of Mirics.

SDRPlay is the trading name of SDRPlay Limited a company registered in England #
09035244.

Mirics is the trading name of Mirics Limited a company registered in England #
05046393.
```

Condition 3 is the reason nothing in this project's documentation says or implies that
SDRplay endorses it.
