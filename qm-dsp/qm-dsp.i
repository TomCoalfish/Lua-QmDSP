%module qm_dsp
%{
#include <cmath>
#include "base/KaiserWindow.h"
#include "base/SincWindow.h"
#include "base/Pitch.h"
#include "dsp/chromagram/Chromagram.h"
#include "dsp/chromagram/ConstantQ.h"
#include "dsp/keydetection/GetKeyMode.h"
#include "dsp/mfcc/MFCC.h"
#include "dsp/onsets/DetectionFunction.h"
#include "dsp/onsets/PeakPicking.h"
#include "dsp/phasevocoder/PhaseVocoder.h"
#include "dsp/rateconversion/Decimator.h"
#include "dsp/rateconversion/DecimatorB.h"
#include "dsp/rateconversion/Resampler.h"
#include "dsp/rhythm/BeatSpectrum.h"
#include "dsp/segmentation/ClusterMeltSegmenter.h"
#include "dsp/segmentation/Segmenter.h"
#include "dsp/segmentation/cluster_segmenter.h"
#include "dsp/segmentation/segment.h"
#include "dsp/signalconditioning/DFProcess.h"
#include "dsp/signalconditioning/FiltFilt.h"
#include "dsp/signalconditioning/Filter.h"
#include "dsp/signalconditioning/Framer.h"
#include "dsp/tempotracking/DownBeat.h"
#include "dsp/tempotracking/TempoTrack.h"
#include "dsp/tempotracking/TempoTrackV2.h"
#include "dsp/tonal/ChangeDetectionFunction.h"
#include "dsp/tonal/TCSgram.h"
#include "dsp/tonal/TonalEstimator.h"
#include "dsp/transforms/DCT.h"
#include "dsp/transforms/FFT.h"
#include "dsp/wavelet/Wavelet.h"
#include "hmm/hmm.h"
#include "maths/Correlation.h"
#include "maths/CosineDistance.h"
#include "maths/KLDivergence.h"
#include "maths/MathUtilities.h"
#include "maths/Polyfit.h"
%}


%include "stdint.i"
%include "std_complex.i"
%include "std_vector.i"
%include "std_math.i"

// you can't use this from swig
struct KaiserParameters {
    int length;
    double beta;
};

class KaiserWindow
{
public:
    

    KaiserWindow(KaiserParameters p);

    static KaiserWindow byTransitionWidth(double attenuation,
                                          double transition);

    static KaiserWindow byBandwidth(double attenuation,
                                    double bandwidth,
                                    double samplerate);

    static KaiserParameters parametersForTransitionWidth(double attenuation,
                                                   double transition);

    static KaiserParameters parametersForBandwidth(double attenuation,
                                             double bandwidth,
                                             double samplerate);

    int getLength() const;
    const double *getWindow();
    void cut(double *src) const;
    void cut(const double *src, double *dst) const;
};

class SincWindow
{
public:

    SincWindow(int length, double p);

    int getLength() const;
    const double *getWindow() const;
    void cut(double *src) const;
    void cut(const double *src, double *dst) const;
};

class Pitch
{
public:
    static float getFrequencyForPitch(int midiPitch,
                                      float centsOffset = 0,
                                      float concertA = 440.0);

    static int getPitchForFrequency(float frequency,
                                    float *centsOffsetReturn = 0,
                                    float concertA = 440.0);
};


enum WindowType {
    RectangularWindow,
    BartlettWindow,
    HammingWindow,
    HanningWindow,
    BlackmanWindow,
    BlackmanHarrisWindow,
    FirstWindow = RectangularWindow,
    LastWindow = BlackmanHarrisWindow
};


/**
 * Various shaped windows for sample frame conditioning, including
 * cosine windows (Hann etc) and triangular and rectangular windows.
 */
template <typename T>
class Window
{
public:

    Window(WindowType type, int size);
    Window(const Window &w);
    Window &operator=(const Window &w);
    virtual ~Window();
    
    void cut(T *src) const;
    void cut(const T *src, T *dst) const;

    WindowType getType() const;
    int getSize() const;

    std::vector<T> getWindowData() const;
};


struct ChromaConfig {
    double FS;
    double min;
    double max;
    int BPO;
    double CQThresh;
    MathUtilities::NormaliseType normalise;
};

class Chromagram 
{
public: 
    Chromagram( ChromaConfig Config );
    ~Chromagram();

    double *process(const double *data);
    double *process(const double *real, const double *imag);   
    void unityNormalise(double* src);
    double kabs( double real, double imag );    
    int getK() { return m_uK;}
    int getFrameSize() { return m_frameSize; }
    int getHopSize()   { return m_hopSize; }
    
};

struct CQConfig {
    double FS;         // samplerate
    double min;        // minimum frequency
    double max;        // maximum frequency
    int BPO;           // bins per octave
    double CQThresh;   // threshold
};

class ConstantQ
{
public:
    ConstantQ(CQConfig config);
    ~ConstantQ();

    void process(const double* FFTRe, const double* FFTIm,
                 double* CQRe, double* CQIm);

    double* process(const double* FFTData);
    void sparsekernel();

    double getQ();
    int getK();
    int getFFTLength();
    int getHop();
};


struct KeyModeConfig {
        double sampleRate;
        float tuningFrequency;
        double hpcpAverage;
        double medianAverage;
        int frameOverlapFactor; // 1 = none (default, fast, but means
                                // we skip a fair bit of input data);
                                // 8 = normal chroma overlap
        int decimationFactor;

        KeyModeConfig(double _sampleRate, float _tuningFrequency);
    };
    

class GetKeyMode  
{
public:
    
    GetKeyMode(KeyModeConfig config);

    virtual ~GetKeyMode();

    /**
     * Process a single time-domain input sample frame of length
     * getBlockSize(). Successive calls should provide overlapped data
     * with an advance of getHopSize() between frames.
     *
     * Return a key index in the range 0-24, where 0 indicates no key
     * detected, 1 is C major, and 13 is C minor.
     */
    int process(double *pcmData);

    /**
     * Return a pointer to an internal 24-element array containing the
     * correlation of the chroma vector generated in the last
     * process() call against the stored key profiles for the 12 major
     * and 12 minor keys, where index 0 is C major and 12 is C minor.
     */
    double *getKeyStrengths();

    int getBlockSize();
    int getHopSize();
};

struct MFCCConfig {
    int FS;
    int fftsize;
    int nceps;
    double logpower;
    bool want_c0;
    WindowType window;
    MFCCConfig(int _FS) :
        FS(_FS), fftsize(2048), nceps(19),
        logpower(1.0), want_c0(true), window(HammingWindow) { }
};

class MFCC
{
public:
    MFCC(MFCCConfig config);
    virtual ~MFCC();

    int process(const double *inframe, double *outceps);
    int process(const double *real, const double *imag, double *outceps);
    int getfftlength() const;
};

/*
#define DF_HFC (1)
#define DF_SPECDIFF (2)
#define DF_PHASEDEV (3)
#define DF_COMPLEXSD (4)
#define DF_BROADBAND (5)
*/

struct DFConfig{
    int stepSize; // DF step in samples
    int frameLength; // DF analysis window - usually 2*step. Must be even!
    int DFType; // type of detection function ( see defines )
    double dbRise; // only used for broadband df (and required for it)
    bool adaptiveWhitening; // perform adaptive whitening
    double whiteningRelaxCoeff; // if < 0, a sensible default will be used
    double whiteningFloor; // if < 0, a sensible default will be used
};

class DetectionFunction  
{
public:
    double* getSpectrumMagnitude();
    DetectionFunction( DFConfig config );
    virtual ~DetectionFunction();

    double processTimeDomain(const double* samples);
    double processFrequencyDomain(const double* reals, const double* imags);
};


struct PPWinThresh
{
    int pre;
    int post;

    PPWinThresh(int x, int y);
};

struct QFitThresh
{
    double a;
    double b;
    double c;

    QFitThresh(double x, double y, double z);
};

struct PPickParams
{
    int length; // detection function length
    double tau; // time resolution of the detection function
    int alpha; // alpha-norm parameter
    double cutoff;// low-pass filter cutoff freq
    int LPOrd; // low-pass filter order
    double* LPACoeffs; // low-pass filter denominator coefficients
    double* LPBCoeffs; // low-pass filter numerator coefficients
    PPWinThresh WinT;// window size in frames for adaptive thresholding [pre post]:
    QFitThresh QuadThresh;
    float delta; // delta threshold used as an offset when computing the smoothed detection function

    PPickParams();
};

class PeakPicking  
{
public:
    PeakPicking( PPickParams Config );
    virtual ~PeakPicking();
        
    void process( double* src, int len, std::vector<int> &onsets  );
};


class PhaseVocoder  
{
public:
    PhaseVocoder(int size, int hop);
    virtual ~PhaseVocoder();

    void processTimeDomain(const double *src,
                           double *mag, double *phase, double *unwrapped);

    void processFrequencyDomain(const double *reals, const double *imags,
                                double *mag, double *phase, double *unwrapped);

    void reset();

};


class Decimator  
{
public:

    Decimator(int inLength, int decFactor);
    virtual ~Decimator();

    void process( const double* src, double* dst );
    void process( const float* src, float* dst );

    int getFactor() const;
    static int getHighestSupportedFactor();

    void resetFilter();
};


class DecimatorB
{
public:
    void process( const double* src, double* dst );
    void process( const float* src, float* dst );

    DecimatorB(int inLength, int decFactor);
    virtual ~DecimatorB();
    int getFactor() const;
};

class Resampler
{
public:

    Resampler(int sourceRate, int targetRate);
    Resampler(int sourceRate, int targetRate, double snr, double bandwidth);
    virtual ~Resampler();

    int process(const double *src, double *dst, int n);
    std::vector<double> process(const double *src, int n);
    int getLatency() const;
    static std::vector<double> resample(int sourceRate, int targetRate, const double *data, int n);
};

class BeatSpectrum
{
public:
    BeatSpectrum();
    ~BeatSpectrum();
    std::vector<double> process(const std::vector<std::vector<double> > &inmatrix);

};


class ClusterMeltSegmenterParams
// defaults are sensible for 11025Hz with 0.2 second hopsize
{
public:
    ClusterMeltSegmenterParams();
    feature_types featureType;
    double hopSize;     // in secs
    double windowSize;  // in secs
    int fmin;
    int fmax;
    int nbins;
    int ncomponents;
    int nHMMStates;
    int nclusters;
    int histogramLength;
    int neighbourhoodLimit;
};



class Segment
{
public:
    int start;           
    int end;
    int type;
};

class Segmentation
{
public:
    int nsegtypes;       
    int samplerate;
    std::vector<Segment> segments;       
};

class Segmenter
{
public:
    Segmenter() {}
    virtual ~Segmenter() {}
    virtual void initialise(int samplerate) = 0;    // must be called before any other methods
    virtual int getWindowsize() = 0;                                // required window size for calls to extractFeatures()
    virtual int getHopsize() = 0;                                   // required hop size for calls to extractFeatures()
    virtual void extractFeatures(const double* samples, int nsamples) = 0;
    virtual void segment() = 0;                                             // call once all the features have been extracted
    virtual void segment(int m) = 0;                                // specify desired number of segment-types
    virtual void clear();
    const Segmentation& getSegmentation() const;
};

class ClusterMeltSegmenter : public Segmenter
{
public:
    ClusterMeltSegmenter(ClusterMeltSegmenterParams params);
    virtual ~ClusterMeltSegmenter();
    virtual void initialise(int samplerate);
    virtual int getWindowsize();
    virtual int getHopsize();
    virtual void extractFeatures(const double* samples, int nsamples);
    void setFeatures(const std::vector<std::vector<double> >& f);         // provide the features yourself
    virtual void segment();             // segment into default number of segment-types
    void segment(int m);                // segment into m segment-types
    int getNSegmentTypes();
};

void cluster_melt(double *h,            /* normalised histograms, as a vector in row major order */
                  int m,                        /* number of dimensions (i.e. histogram bins) */
                  int n,                        /* number of histograms */
                  double *Bsched,       /* inverse temperature schedule */
                  int t,                        /* length of schedule */
                  int k,                        /* number of clusters */
                  int l,                        /* neighbourhood limit (supply zero to use default value) */
                  int *c                        /* sequence of cluster assignments */
    );

typedef struct segment_t
{
    long start;                     /* in samples */
    long end;
    int type;
} segment_t;

typedef struct segmentation_t
{
    int nsegs; /* number of segments */
    int nsegtypes; /* number of segment types, so possible types are {0,1,...,nsegtypes-1} */
    int samplerate;
    segment_t* segments;
} segmentation_t;

typedef enum 
{ 
    FEATURE_TYPE_UNKNOWN = 0, 
    FEATURE_TYPE_CONSTQ = 1, 
    FEATURE_TYPE_CHROMA = 2,
    FEATURE_TYPE_MFCC = 3
} feature_types;

/* applies MPEG-7 normalisation to constant-Q features,
   storing normalised envelope (norm) in last feature dimension */
void mpeg7_constq(double** features, int nframes, int ncoeff);

/* converts constant-Q features to normalised chroma */
void cq2chroma(double** cq, int nframes, int ncoeff, int bins, double** chroma);

void create_histograms(int* x, int nx, int m, int hlen, double* h);

void cluster_segment(int* q, double** features, int frames_read,
                     int feature_length, int nHMM_states, 
                     int histogram_length, int nclusters,
                     int neighbour_limit);

void constq_segment(int* q, double** features, int frames_read,
                    int bins, int ncoeff, int feature_type, 
                    int nHMM_states, int histogram_length,
                    int nclusters, int neighbour_limit);


struct DFProcConfig
{
    int length; 
    int LPOrd; 
    double *LPACoeffs; 
    double *LPBCoeffs; 
    int winPre;
    int winPost; 
    double AlphaNormParam;
    bool isMedianPositive;
    float delta; //delta threshold used as an offset when computing the smoothed detection function
    DFProcConfig();
};

class DFProcess  
{
public:
    DFProcess( DFProcConfig Config );
    virtual ~DFProcess();
    void process( double* src, double* dst );        
};

struct FilterParameters {
    std::vector<double> a;
    std::vector<double> b;
};



class FiltFilt  
{
public:
    FiltFilt(FilterParameters);
    virtual ~FiltFilt();

    void process(const double *const src,
                 double *const  dst,
                 const int length);
};



class Filter
{
public:
    
    
    Filter(FilterParameters params);
    
    ~Filter();

    void reset();

    void process(const double *const in,
                 double *const out,
                 const int n);

    int getOrder() const;    
};

class Framer  
{
public:
    Framer();
    virtual ~Framer();

    void setSource(double* src, int64_t length);
    void configure(int frameLength, int hop);
    
    int getMaxNoFrames();
    void getFrame(double* dst);

    void resetCounters();
};

class DownBeat
{
public:
    DownBeat(float originalSampleRate,
             size_t decimationFactor,
             size_t dfIncrement);
    ~DownBeat();

    void setBeatsPerBar(int bpb);
    void findDownBeats(const float *audio, // downsampled
                       size_t audioLength, // after downsampling
                       const std::vector<double> &beats,
                       std::vector<int> &downbeats);
    void getBeatSD(std::vector<double> &beatsd) const;
    void pushAudioBlock(const float *audio);
    const float *getBufferedAudio(size_t &length) const;
    void resetAudioBuffer();
};


struct WinThresh
{
    int pre;
    int post;
};

struct TTParams
{
    int winLength; //Analysis window length
    int lagLength; //Lag & Stride size
    int alpha; //alpha-norm parameter
    int LPOrd; // low-pass Filter order
    double* LPACoeffs; //low pass Filter den coefficients
    double* LPBCoeffs; //low pass Filter num coefficients
    WinThresh WinT;//window size in frames for adaptive thresholding [pre post]:
};


class TempoTrack  
{
public:
    TempoTrack( TTParams Params );
    virtual ~TempoTrack();

    std::vector<int> process( std::vector <double> DF,
                              std::vector <double> *tempoReturn = 0);
};


class TempoTrackV2
{
public:

    /**
     * Construct a tempo tracker that will operate on beat detection
     * function data calculated from audio at the given sample rate
     * with the given frame increment.
     *
     * Currently the sample rate and increment are used only for the
     * conversion from beat frame location to bpm in the tempo array.
     */
    TempoTrackV2(float sampleRate, int dfIncrement);
    ~TempoTrackV2();

    // Returned beat periods are given in df increment units; inputtempo and tempi in bpm
    void calculateBeatPeriod(const std::vector<double> &df,
                             std::vector<double> &beatPeriod,
                             std::vector<double> &tempi);

    // Returned beat periods are given in df increment units; inputtempo and tempi in bpm
    // MEPD 28/11/12 Expose inputtempo and constraintempo parameters
    // Note, if inputtempo = 120 and constraintempo = false, then functionality is as it was before
    void calculateBeatPeriod(const std::vector<double> &df,
                             std::vector<double> &beatPeriod,
                             std::vector<double> &tempi,
                             double inputtempo, bool constraintempo);

    // Returned beat positions are given in df increment units
    void calculateBeats(const std::vector<double> &df,
                        const std::vector<double> &beatPeriod,
                        std::vector<double> &beats);

    // Returned beat positions are given in df increment units
    // MEPD 28/11/12 Expose alpha and tightness parameters
    // Note, if alpha = 0.9 and tightness = 4, then functionality is as it was before
    void calculateBeats(const std::vector<double> &df,
                        const std::vector<double> &beatPeriod,
                        std::vector<double> &beats,
                        double alpha, double tightness);

};


struct ChangeDFConfig
{
    int smoothingWidth;
};

class ChangeDetectionFunction
{
public:
    ChangeDetectionFunction(ChangeDFConfig);
    ~ChangeDetectionFunction();
    ChangeDistance process(const TCSGram& rTCSGram);
};


class TCSGram
{
public: 
    TCSGram();
    ~TCSGram();
    void getTCSVector(int, TCSVector&) const;
    void addTCSVector(const TCSVector&);
    long getTime(size_t) const;
    long getDuration() const;
    void printDebug();
    int getSize() const;
    void reserve(size_t uSize);
    void clear();
    void setFrameDuration(const double dFrameDurationMS);
    void setNumBins(const unsigned int uNumBins);
    //void normalize();
};


class ChromaVector 
{
public:
    ChromaVector(size_t uSize = 12) : std::valarray<double>();
    virtual ~ChromaVector();
    void printDebug();
    void normalizeL1();
    void clear();

    %extend {
        double __getitem__(size_t i) { return (*$self)[i]; }
        void   __setitem__(size_t i, double val) { (*$self)[i] = val; }
    }
};

class TCSVector
{
public:
    TCSVector() : std::valarray<double>();
    virtual ~TCSVector();
    void printDebug();    
    double magnitude() const;

    %extend {
        double __getitem__(size_t i) { return (*$self)[i]; }
        void   __setitem__(size_t i, double value) { (*$self)[i] = value; }
    }
};

class TonalEstimator
{
public:
    TonalEstimator();
    virtual ~TonalEstimator();
    TCSVector transform2TCS(const ChromaVector& rVector);
};



class DCT
{
public:
    
    DCT(int n);
    ~DCT();

    void forward(const double *in, double *out);
    void forwardUnitary(const double *in, double *out);
    void inverse(const double *in, double *out);
    void inverseUnitary(const double *in, double *out);

};

class FFT  
{
public:
    /**
     * Construct an FFT object to carry out complex-to-complex
     * transforms of size nsamples. nsamples does not have to be a
     * power of two.
     */
    FFT(int nsamples);
    ~FFT();

    /**
     * Carry out a forward or inverse transform (depending on the
     * value of inverse) of size nsamples, where nsamples is the value
     * provided to the constructor above.
     *
     * realIn and (where present) imagIn should contain nsamples each,
     * and realOut and imagOut should point to enough space to receive
     * nsamples each.
     *
     * imagIn may be NULL if the signal is real, but the other
     * pointers must be valid.
     *
     * The inverse transform is scaled by 1/nsamples.
     */
    void process(bool inverse,
                 const double *realIn, const double *imagIn,
                 double *realOut, double *imagOut);
    
};

class FFTReal
{
public:
    /**
     * Construct an FFT object to carry out real-to-complex transforms
     * of size nsamples. nsamples does not have to be a power of two,
     * but it does have to be even. (Use the complex-complex FFT above
     * if you need an odd FFT size. This constructor will throw
     * std::invalid_argument if nsamples is odd.)
     */
    FFTReal(int nsamples);
    ~FFTReal();

    /**
     * Carry out a forward real-to-complex transform of size nsamples,
     * where nsamples is the value provided to the constructor above.
     *
     * realIn, realOut, and imagOut must point to (enough space for)
     * nsamples values. For consistency with the FFT class above, and
     * compatibility with existing code, the conjugate half of the
     * output is returned even though it is redundant.
     */
    void forward(const double *realIn,
                 double *realOut, double *imagOut);

    /**
     * Carry out a forward real-to-complex transform of size nsamples,
     * where nsamples is the value provided to the constructor
     * above. Return only the magnitudes of the complex output values.
     *
     * realIn and magOut must point to (enough space for) nsamples
     * values. For consistency with the FFT class above, and
     * compatibility with existing code, the conjugate half of the
     * output is returned even though it is redundant.
     */
    void forwardMagnitude(const double *realIn, double *magOut);

    /**
     * Carry out an inverse real transform (i.e. complex-to-real) of
     * size nsamples, where nsamples is the value provided to the
     * constructor above.
     *
     * realIn and imagIn should point to at least nsamples/2+1 values;
     * if more are provided, only the first nsamples/2+1 values of
     * each will be used (the conjugate half will always be deduced
     * from the first nsamples/2+1 rather than being read from the
     * input data).  realOut should point to enough space to receive
     * nsamples values.
     *
     * The inverse transform is scaled by 1/nsamples.
     */
    void inverse(const double *realIn, const double *imagIn,
                 double *realOut);

};    


class Wavelet
{
public:
    enum Type {
        Haar = 0,
        Daubechies_2,
        Daubechies_3,
        Daubechies_4,
        Daubechies_5,
        Daubechies_6,
        Daubechies_7,
        Daubechies_8,
        Daubechies_9,
        Daubechies_10,
        Daubechies_20,
        Daubechies_40,
        Symlet_2,
        Symlet_3,
        Symlet_4,
        Symlet_5,
        Symlet_6,
        Symlet_7,
        Symlet_8,
        Symlet_9,
        Symlet_10,
        Symlet_20,
        Symlet_30,
        Coiflet_1,
        Coiflet_2,
        Coiflet_3,
        Coiflet_4,
        Coiflet_5,
        Biorthogonal_1_3,
        Biorthogonal_1_5,
        Biorthogonal_2_2,
        Biorthogonal_2_4,
        Biorthogonal_2_6,
        Biorthogonal_2_8,
        Biorthogonal_3_1,
        Biorthogonal_3_3,
        Biorthogonal_3_5,
        Biorthogonal_3_7,
        Biorthogonal_3_9,
        Biorthogonal_4_4,
        Biorthogonal_5_5,
        Biorthogonal_6_8,
        Meyer,

        LastType = Meyer
    };

    static std::string getWaveletName(Type);

    static void createDecompositionFilters(Type,
                                           std::vector<double> &lpd,
                                           std::vector<double> &hpd);
};



typedef struct _model_t {
    int N;          /* number of states */
    double* p0;     /* initial probs */
    double** a;     /* transition probs */
    int L;          /* dimensionality of data */
    double** mu;    /* state means */
    double** cov;   /* covariance, tied between all states */
} model_t;

void hmm_train(double** x, int T, model_t* model); /* with scaling */

void forward_backwards(double*** xi, double** gamma,
                       double* loglik, double* loglik1, double* loglik2,
                       int iter, int N, int T,
                       double* p0, double** a, double** b);
    
void baum_welch(double* p0, double** a, double** mu, double** cov,
                int N, int T, int L, double** x, double*** xi, double** gamma);

void viterbi_decode(double** x, int T, model_t* model, int* q); /* using logs */

model_t* hmm_init(double** x, int T, int L, int N);
void hmm_close(model_t* model);
    
void invert(double** cov, int L, double** icov, double* detcov); /* uses LAPACK */
    
double gauss(double* x, int L, double* mu, double** icov,
             double detcov, double* y, double* z);
    
double loggauss(double* x, int L, double* mu, double** icov,
                double detcov, double* y, double* z);
    
void hmm_print(model_t* model);


class Correlation  
{
public:
    Correlation();
    virtual ~Correlation();

    void doAutoUnBiased( double* src, double* dst, int length );
};

class CosineDistance
{
public:
    CosineDistance();
    ~CosineDistance();

    double distance(const std::vector<double> &v1,
                    const std::vector<double> &v2);
};

class KLDivergence
{
public:
    KLDivergence();
    ~KLDivergence();

    /**
     * Calculate a symmetrised Kullback-Leibler divergence of Gaussian
     * models based on mean and variance vectors.  All input vectors
     * must be of equal size.
     */
    double distanceGaussian(const std::vector<double> &means1,
                            const std::vector<double> &variances1,
                            const std::vector<double> &means2,
                            const std::vector<double> &variances2);

    /**
     * Calculate a Kullback-Leibler divergence of two probability
     * distributions.  Input vectors must be of equal size.  If
     * symmetrised is true, the result will be the symmetrised
     * distance (equal to KL(d1, d2) + KL(d2, d1)).
     */
    double distanceDistribution(const std::vector<double> &d1,
                                const std::vector<double> &d2,
                                bool symmetrised);
};




class MathUtilities  
{
public: 

    /**
     * Round x to the nearest integer.
     */
    static double round( double x );

    /**
     * Return through min and max pointers the highest and lowest
     * values in the given array of the given length.
     */
    static void getFrameMinMax( const double* data, int len,
                                double* min, double* max );

    /**
     * Return the mean of the given array of the given length.
     */
    static double mean( const double* src, int len );

    /**
     * Return the mean of the subset of the given vector identified by
     * start and count.
     */
    static double mean( const std::vector<double> &data,
                        int start, int count );
    
    /**
     * Return the sum of the values in the given array of the given
     * length.
     */
    static double sum( const double* src, int len );

    /**
     * Return the median of the values in the given array of the given
     * length. If the array is even in length, the returned value will
     * be half-way between the two values adjacent to median.
     */
    static double median( const double* src, int len );

    /**
     * The principle argument function. Map the phase angle ang into
     * the range [-pi,pi).
     */
    static double princarg( double ang );

    /**
     * Floating-point division modulus: return x % y.
     */
    static double mod( double x, double y);

    /**
     * The alpha norm is the alpha'th root of the mean alpha'th power
     * magnitude. For example if alpha = 2 this corresponds to the RMS
     * of the input data, and when alpha = 1 this is the mean
     * magnitude.
     */
    static void getAlphaNorm(const double *data, int len, int alpha, double* ANorm);

    /**
     * The alpha norm is the alpha'th root of the mean alpha'th power
     * magnitude. For example if alpha = 2 this corresponds to the RMS
     * of the input data, and when alpha = 1 this is the mean
     * magnitude.
     */
    static double getAlphaNorm(const std::vector <double> &data, int alpha );

    enum NormaliseType {
        NormaliseNone,
        NormaliseUnitSum,
        NormaliseUnitMax
    };

    static void normalise(double *data, int length,
                          NormaliseType n = NormaliseUnitMax);

    static void normalise(std::vector<double> &data,
                          NormaliseType n = NormaliseUnitMax);

    /**
     * Calculate the L^p norm of a vector. Equivalent to MATLAB's
     * norm(data, p).
     */
    static double getLpNorm(const std::vector<double> &data,
                            int p);

    /**
     * Normalise a vector by dividing through by its L^p norm. If the
     * norm is below the given threshold, the unit vector for that
     * norm is returned. p may be 0, in which case no normalisation
     * happens and the data is returned unchanged.
     */
    static std::vector<double> normaliseLp(const std::vector<double> &data,
                                           int p,
                                           double threshold = 1e-6);
    
    /**
     * Threshold the input/output vector data against a moving-mean
     * average filter.
     */
    static void adaptiveThreshold(std::vector<double> &data);

    static void circShift( double* data, int length, int shift);

    static int getMax( double* data, int length, double* max = 0 );
    static int getMax( const std::vector<double> &data, double* max = 0 );
    static int compareInt(const void * a, const void * b);

    /** 
     * Return true if x is 2^n for some integer n >= 0.
     */
    static bool isPowerOfTwo(int x);

    /**
     * Return the next higher integer power of two from x, e.g. 1300
     * -> 2048, 2048 -> 2048.
     */
    static int nextPowerOfTwo(int x);
    static int previousPowerOfTwo(int x);
    static int nearestPowerOfTwo(int x);
    static double factorial(int x); // returns double in case it is large
    static int gcd(int a, int b);
};



class TPolyFit
{
    typedef vector<vector<double> > Matrix;
public:

    static double PolyFit2 (const vector<double> &x,  // does the work
                            const vector<double> &y,
                            vector<double> &coef);   

private:
    TPolyFit();             
};

// some utility functions

struct NSUtility
{
    static void swap(double &a, double &b);
    // fills a vector with zeros.
    static void zeroise(vector<double> &array, int n);
    // fills a vector with zeros.
    static void zeroise(vector<int> &array, int n);
    
    // fills a (m by n) matrix with zeros.
    static void zeroise(vector<vector<double> > &matrix, int m, int n);
    
    // fills a (m by n) matrix with zeros.
    static void zeroise(vector<vector<int> > &matrix, int m, int n);
    
    static double sqr(const double &x);
};
