# CGR Algorithm Rewrite - Complete Redesign for Keyboard Context

## CRITICAL INSIGHT
The original CGR algorithm was designed for **free-form drawing on blank canvas**. We need to **completely rewrite the equation** for **keyboard-specific constraints** rather than forcing CGR to work inappropriately.

## CURRENT CALIBRATION RESULTS (0% Accuracy)
Despite extensive optimization, CGR consistently fails:

```
KNOWN: Template 1787px vs User 2217px = good match → gets "express" (wrong)
SECTOR: Template 1813px vs User 1996px = 91% match → gets "assist" (wrong)  
COMBAT: Template 2088px vs User 2146px = 97% match → gets "bullshit" (wrong)
BAND: Template 1689px vs User 1768px = 95% match → gets "message" (wrong)
ANALYSIS: No results at all despite good template
APART: Template 2284px vs User 2481px = 92% match → gets "asleep" (wrong)
```

### KEY INSIGHTS FROM FAILURES:
1. **Length matching excellent** (90-97% similarity) but **wrong words**
2. **Coordinate alignment good** (0.7-0.9 scores) but **irrelevant**
3. **Templates geometrically correct** but **algorithm inappropriate**
4. **Total trace length** often nearly matched template (critical insight!)

## PROBLEM ANALYSIS

### Original CGR Context vs Keyboard Context:
- **Original**: Free drawing anywhere, any size → **High variance**
- **Keyboard**: "Follow the dotted line" → **Constrained paths**
- **Original**: Shape recognition → **Euclidean + turning angle**
- **Keyboard**: Key sequence matching → **Proximity + letter order**

### Why CGR Fails for Keyboards:
1. **Designed for shape recognition**, not **letter sequence recognition**
2. **Distance metrics inappropriate** for constrained key paths
3. **Vocabulary scale** (3000 words) vs original (50 templates)
4. **Template matching** vs **key proximity matching**

## NEW ALGORITHM DESIGN

### Core Equation (Bayesian Framework):
```
P(word | swipe) ∝ P(swipe | word) × P(word)
```

### Component Redesign:

#### 1. **Prior Probability P(word)**
- **Language model**: Word frequency in dictionary
- **N-gram context**: Previous words for prediction
- **User adaptation**: Personal word usage patterns

#### 2. **Likelihood P(swipe | word) - COMPLETELY NEW**
Replace CGR distance with **keyboard-specific cost function**:

```
Cost = α × ProximityPenalty + β × MissingKeyPenalty + γ × ExtraKeyPenalty + δ × OrderPenalty + ε × StartPointWeight
```

##### **Proximity to Keys**: 
- Measure distance from swipe path to actual key centers
- Not template line matching - **key zone proximity**

##### **Missing Key Penalty**:
- High cost if swipe misses required letter key zones
- Each missed letter dramatically reduces probability

##### **Extra Key Penalty**: 
- Lower cost for passing over non-template keys
- Penalize but don't eliminate for incidental key touches

##### **Letter Order Enforcement**:
- Swipe must pass near keys in correct sequence
- Out-of-order key touching heavily penalized

##### **Start Point Emphasis** (Your Insight):
- **Higher weight for start point accuracy** (users begin precisely)
- **Lower weight for end point** (users end sloppily)
- **ε > δ** in equation weighting

### Algorithm Flow:
1. **Candidate Generation**: Find words whose letters are near swipe path
2. **Key Proximity Analysis**: Calculate distances to each letter key
3. **Sequence Validation**: Verify correct letter order
4. **Cost Calculation**: Apply keyboard-specific penalty function
5. **Bayesian Ranking**: Combine with language model priors

## EXISTING CODE ASSETS (EXTENSIVE INFRASTRUCTURE AVAILABLE)

### Prediction Infrastructure:
- **WordPredictor.java**: Language model and word frequency ✅
- **BigramModel.java**: N-gram context prediction with P(word | previous_word) ✅
- **NgramModel.java**: Advanced language modeling ✅
- **DTWPredictor.java**: Dynamic Time Warping for swipe-to-word matching ✅
- **EnhancedWordPredictor.java**: Enhanced prediction capabilities ✅
- **DictionaryManager.java**: Word database access ✅
- **UserAdaptationManager.java**: Personal usage patterns ✅

### Specialized Components:
- **SwipePruner**: Path optimization ✅
- **GaussianKeyModel**: Key zone modeling ✅ 
- **SwipeWeightConfig**: Configurable algorithm weights ✅
- **SwipeCalibrationActivity.java**: Comprehensive testing framework ✅

### Template Generation Assets:
- **Dynamic keyboard layout**: Actual key positions ✅
- **Coordinate system**: Screen-space alignment ✅
- **Key center calculations**: Accurate positioning ✅
- **Real keyboard dimensions**: User height settings ✅

## IMPLEMENTATION PLAN

### Phase 1: Core Algorithm Rewrite
1. **Abandon CGR distance metrics** entirely
2. **Implement key proximity matching**
3. **Add letter sequence validation**
4. **Create keyboard-specific cost function**

### Phase 2: Integration
1. **Combine with existing language models**
2. **Integrate user adaptation data**
3. **Maintain real-time prediction capability**
4. **Preserve settings infrastructure**

### Phase 3: Optimization
1. **Parameter tuning for keyboard context**
2. **Performance optimization for 3000-word vocabulary**
3. **User-configurable weights and penalties**

## IMPLEMENTATION STATUS

### COMPLETED:
1. **Analysis and planning documented in refactor.md** ✅
2. **KeyboardSwipeRecognizer.java created** with Bayesian framework ✅
3. **Infrastructure assessment** - extensive existing code available ✅
4. **Algorithm framework defined** with keyboard-specific components ✅

### IN PROGRESS:
1. **Implementing key proximity detection**
2. **Letter sequence validation with order enforcement**
3. **Start point emphasis weighting system**
4. **Integration with existing BigramModel and NgramModel**

### NEXT STEPS:
1. **Complete KeyboardSwipeRecognizer implementation**
2. **Replace CGR calls with new algorithm**
3. **Test against current calibration data**
4. **Integrate existing prediction infrastructure**
5. **Update refactor.md after each major change** 

### REMEMBER TO UPDATE REFACTOR.MD:
- Document each implementation step
- Record test results and calibration improvements  
- Track integration with existing prediction components
- Note parameter tuning and optimization discoveries

The CGR approach was fundamentally wrong for keyboard context. The new approach should achieve dramatically better accuracy by focusing on **key proximity** and **letter sequence matching** rather than **abstract shape recognition**.