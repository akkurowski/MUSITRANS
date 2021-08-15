<span style="font-size:14pt;color:##820000;">*This is a developer branch of the project - it is recomended to use a stable version from the `main` branch*</span>
<br/>

<p align="center"><img src="https://github.com/akkurowski/MUSITRANS/blob/main/img/musitrans_logo.png?raw=true" alt="logo" width="1000"/></p>
<p align="center" style="font-size:x-large; font-style: italic">a one hobbyists project for automatic music transcription</p>

<p>    This is my after-hours project which purpose is to analyse and visualize spectrograms of music. It also makes it possible to extract melodic lines from processed music pieces. I developed it with an intention of employing it for a mass-generation of MIDI files which can be then fed as an input to some algorithmic composition software. For now - it is a simple, rather suboptimal, but probably somehow useful little Python script. If you find a use for it in your own project - feel free to use it! I would also be very glad if you let me know about it - feel free to contact me in such case!</p>

<p>The software can calculate oversampled amplitude spectra, like the one below:</p>
<p align="center"><img src="https://github.com/akkurowski/MUSITRANS/blob/main/img/spect_cut.png?raw=true" alt="spectrogram" width="600"/></p>

<p>It is also capable of computing estimated melodic lines with the use of a simple heuristic algorithm employing chroma features-based salience calculation and simulated annealing for the trajectory smoothing:</p>
<p align="center"><img src="https://github.com/akkurowski/MUSITRANS/blob/main/img/detected_voices_cut.png?raw=true" alt="tracing results" width="600"/></p>

To run the algorithm, just put your audio files in the *_data/input_signals* The following extensions are allowed: *.wav*, *.mp3*, and *.m4a*, however other formats can be specified in the *settings.ini* file. Next just run the *musitrans.py* and choose adequate options from the CLI.

Keep in mind, that for processing input files other than WAVE files you have to install <a href="https://www.ffmpeg.org">FFmpeg</a> and add it to the *Path* system environment variable, so it can be called from the console by the MUSITRANS script.

<p align="center">*****</p>

<p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>
