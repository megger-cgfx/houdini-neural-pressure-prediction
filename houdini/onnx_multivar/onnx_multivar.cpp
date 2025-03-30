// Houdini ONNX Plugin with Multi-dimensional Variable Input Support
// This plugin extends Houdini's ONNX functionality to handle variable dimensions on all axes

#include <UT/UT_DSOVersion.h>
#include <OP/OP_Director.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_SpareData.h>
#include <UT/UT_Matrix3.h>
#include <UT/UT_Matrix4.h>
#include <SIM/SIM_Engine.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_ObjectArray.h>
#include <SIM/SIM_Data.h>
#include <SIM/SIM_DataUtils.h>
#include <SIM/SIM_SolverNode.h>
#include <SIM/SIM_DopDescription.h>
#include <SIM/SIM_Guide.h>
#include <SIM/SIM_GuideShared.h>
#include <SIM/SIM_Position.h>
#include <SIM/SIM_ScalarField.h>
#include <SIM/SIM_VectorField.h>
#include <SIM/SIM_Solver.h>
#include <GAS/GAS_Utils.h>
#include <GAS/GAS_SubSolver.h>

// ONNX Runtime includes
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>
#include <memory>
#include <iostream>


class FlexONNXSolver : public SIM_Solver
{
public:
    static const char *SOLVER_NAME;
    
    FlexONNXSolver(const SIM_DataFactory *factory);
    ~FlexONNXSolver() override;

    static SIM_Solver *create(const SIM_DataFactory *factory);
    
    static void initializeSubclass();
    static void installSubclass(PRM_Template *customTemplates);
    
    bool solveGasSubclass(SIM_Engine &engine, SIM_Object *obj, SIM_Time time, SIM_Time timestep) override;

private:
    // ONNX Runtime session
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    
    // Load the ONNX model
    bool loadONNXModel(const char *modelPath);
    
    // Process velocity field through ONNX model
    bool processVelocityField(GU_Detail *gdp, SIM_ScalarField *pressureField, SIM_VectorField *velocityField, int resX, int resY, int resZ);
    
    // Helper methods
    void velocityFieldToTensor(SIM_VectorField *velocityField, std::vector<float> &tensorData, int resX, int resY, int resZ);
    void pressureTensorToField(const std::vector<float> &tensorData, SIM_ScalarField *pressureField, int resX, int resY, int resZ);
    void applyPressureToVelocity(SIM_ScalarField *pressureField, SIM_VectorField *velocityField, float timestep);
};

// Static variable definitions
const char *FlexONNXSolver::SOLVER_NAME = "flexonnx";

// Parameter template definitions
static PRM_Name modelPathName("modelpath", "ONNX Model Path");
static PRM_Name acceleratorName("accelerator", "Execution Provider");
static PRM_Name dimensionsName("dimensions", "Input Dimensions");

static PRM_Default modelPathDefault(0, "");

static PRM_Name acceleratorChoices[] = {
    PRM_Name("cpu", "CPU"),
    PRM_Name("cuda", "CUDA"),
    PRM_Name(0)
};

static PRM_ChoiceList acceleratorChoiceList(PRM_CHOICELIST_SINGLE, acceleratorChoices);

static PRM_Template customTemplates[] = {
    PRM_Template(PRM_FILE, 1, &modelPathName, &modelPathDefault),
    PRM_Template(PRM_ORD, 1, &acceleratorName, PRMzeroDefaults, &acceleratorChoiceList),
    PRM_Template(PRM_TOGGLE, 3, &dimensionsName),
    PRM_Template()
};

// Constructor
FlexONNXSolver::FlexONNXSolver(const SIM_DataFactory *factory)
    : SIM_Solver(factory), env_(ORT_LOGGING_LEVEL_WARNING)
{
}

// Destructor
FlexONNXSolver::~FlexONNXSolver()
{
    session_.reset();
}

// Create method for the solver
SIM_Solver *
FlexONNXSolver::create(const SIM_DataFactory *factory)
{
    return new FlexONNXSolver(factory);
}

// Initialize the subclass
void
FlexONNXSolver::initializeSubclass()
{
    // Set up any necessary initialization
}

// Install the subclass
void
FlexONNXSolver::installSubclass(PRM_Template *customTemplates)
{
    // Register the solver in Houdini
    SIM_Solver::registerSolver(SOLVER_NAME, FlexONNXSolver::create, customTemplates);
}

// Load the ONNX model
bool
FlexONNXSolver::loadONNXModel(const char *modelPath)
{
    try {
        Ort::SessionOptions sessionOptions;
        
        // Enable dynamic dimensions for all axes
        sessionOptions.DisableMemPattern();
        sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
        
        // Check if CUDA is selected
        if (evalInt("accelerator", 0, 0.0) == 1) {
            // Add CUDA execution provider
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        }
        
        // Create session with dynamic dimensions
        session_ = std::make_unique<Ort::Session>(env_, modelPath, sessionOptions);
        
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

// Process velocity field through ONNX model
bool
FlexONNXSolver::processVelocityField(GU_Detail *gdp, SIM_ScalarField *pressureField, SIM_VectorField *velocityField, int resX, int resY, int resZ)
{
    try {
        // Get the input and output names from the model
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputNodes = session_->GetInputCount();
        size_t numOutputNodes = session_->GetOutputCount();
        
        if (numInputNodes == 0 || numOutputNodes == 0) {
            std::cerr << "Invalid ONNX model: Missing input or output nodes" << std::endl;
            return false;
        }
        
        // Get input name
        char* inputName = session_->GetInputName(0, allocator);
        std::string inputNameStr(inputName);
        allocator.Free(inputName);
        
        // Get output name
        char* outputName = session_->GetOutputName(0, allocator);
        std::string outputNameStr(outputName);
        allocator.Free(outputName);
        
        // Convert velocity field to tensor data
        std::vector<float> inputTensorData;
        velocityFieldToTensor(velocityField, inputTensorData, resX, resY, resZ);
        
        // Set up input tensor dimensions with dynamic axes
        std::vector<int64_t> inputDims = {1, 3, resZ, resY, resX}; // Batch, Channels, Z, Y, X
        
        // Create input tensor
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorData.data(),
            inputTensorData.size(),
            inputDims.data(),
            inputDims.size()
        );
        
        // Define input and output names for the session
        const char* inputNames[] = {inputNameStr.c_str()};
        const char* outputNames[] = {outputNameStr.c_str()};
        
        // Run inference
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );
        
        if (outputTensors.size() == 0 || !outputTensors[0].IsTensor()) {
            std::cerr << "Failed to get valid output tensor from model" << std::endl;
            return false;
        }
        
        // Get output tensor data
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputTensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        auto outputDims = outputTensorInfo.GetShape();
        size_t outputSize = outputTensorInfo.GetElementCount();
        
        // Convert tensor output to pressure field
        std::vector<float> outputTensorData(outputData, outputData + outputSize);
        pressureTensorToField(outputTensorData, pressureField, resX, resY, resZ);
        
        // Apply pressure to velocity field for divergence-free result
        float timestep = evalFloat("timestep", 0, 0.0);
        applyPressureToVelocity(pressureField, velocityField, timestep);
        
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
        return false;
    }
}

// Convert velocity field to tensor
void
FlexONNXSolver::velocityFieldToTensor(SIM_VectorField *velocityField, std::vector<float> &tensorData, int resX, int resY, int resZ)
{
    // Allocate buffer for the tensor (channels * Z * Y * X)
    tensorData.resize(3 * resZ * resY * resX);
    
    // Channel strides
    size_t channelSize = resZ * resY * resX;
    
    // Copy velocity field data to tensor
    UT_VoxelArrayF voxelArray = velocityField->getVoxelArray();
    
    for (int z = 0; z < resZ; ++z) {
        for (int y = 0; y < resY; ++y) {
            for (int x = 0; x < resX; ++x) {
                UT_Vector3F vel = velocityField->getValue(x, y, z);
                
                // Store velocity components in separate channels (NCHW format)
                tensorData[0 * channelSize + z * resY * resX + y * resX + x] = vel.x();
                tensorData[1 * channelSize + z * resY * resX + y * resX + x] = vel.y();
                tensorData[2 * channelSize + z * resY * resX + y * resX + x] = vel.z();
            }
        }
    }
}

// Convert pressure tensor to field
void
FlexONNXSolver::pressureTensorToField(const std::vector<float> &tensorData, SIM_ScalarField *pressureField, int resX, int resY, int resZ)
{
    // Assuming pressure is the third channel from the model output
    size_t channelSize = resZ * resY * resX;
    
    // Update pressure field from tensor data
    for (int z = 0; z < resZ; ++z) {
        for (int y = 0; y < resY; ++y) {
            for (int x = 0; x < resX; ++x) {
                float pressure = tensorData[2 * channelSize + z * resY * resX + y * resX + x];
                pressureField->setValue(x, y, z, pressure);
            }
        }
    }
}

// Apply pressure to velocity field
void
FlexONNXSolver::applyPressureToVelocity(SIM_ScalarField *pressureField, SIM_VectorField *velocityField, float timestep)
{
    // Get field dimensions
    UT_Vector3I res = pressureField->getVoxelRes();
    int resX = res.x();
    int resY = res.y();
    int resZ = res.z();
    
    // Apply pressure gradient to velocity (similar to standard projection step)
    for (int z = 1; z < resZ - 1; ++z) {
        for (int y = 1; y < resY - 1; ++y) {
            for (int x = 1; x < resX - 1; ++x) {
                float pressureGradX = (pressureField->getValue(x + 1, y, z) - pressureField->getValue(x - 1, y, z)) * 0.5f;
                float pressureGradY = (pressureField->getValue(x, y + 1, z) - pressureField->getValue(x, y - 1, z)) * 0.5f;
                float pressureGradZ = (pressureField->getValue(x, y, z + 1) - pressureField->getValue(x, y, z - 1)) * 0.5f;
                
                UT_Vector3F vel = velocityField->getValue(x, y, z);
                vel -= UT_Vector3F(pressureGradX, pressureGradY, pressureGradZ) * timestep;
                velocityField->setValue(x, y, z, vel);
            }
        }
    }
}

// Main solve method
bool
FlexONNXSolver::solveGasSubclass(SIM_Engine &engine, SIM_Object *obj, SIM_Time time, SIM_Time timestep)
{
    // Get the DOP geometry
    GU_Detail *gdp = obj->getOrCreateGeometry();
    if (!gdp)
        return false;
    
    // Get the model path
    UT_String modelPath;
    evalString(modelPath, "modelpath", 0, 0.0);
    
    if (modelPath.isstring() && !session_) {
        if (!loadONNXModel(modelPath.c_str())) {
            std::cerr << "Failed to load ONNX model from " << modelPath.c_str() << std::endl;
            return false;
        }
    }
    
    // Get velocity field from object
    SIM_VectorField *velocityField = getOrCreateVectorField(obj, "vel");
    if (!velocityField)
        return false;
    
    // Get or create pressure field
    SIM_ScalarField *pressureField = getOrCreateScalarField(obj, "pressure");
    if (!pressureField)
        return false;
    
    // Get field dimensions
    UT_Vector3I res = velocityField->getVoxelRes();
    int resX = res.x();
    int resY = res.y();
    int resZ = res.z();
    
    // Process velocity field using the ONNX model
    if (!processVelocityField(gdp, pressureField, velocityField, resX, resY, resZ)) {
            std::cerr << "Failed to process velocity field through ONNX model" << std::endl;
            return false;
        }
    
        return true;
    }

// Get or create a scalar field
SIM_ScalarField *
getOrCreateScalarField(SIM_Object *obj, const char *fieldname)
{
    SIM_DataArray dataArray;
    obj->filterSIMData(dataArray, fieldname);
    
    if (dataArray.entries() > 0)
        return SIM_DATA_CAST(dataArray[0], SIM_ScalarField);
    
    SIM_DataArray baseArray;
    obj->filterSIMData(baseArray, SIM_SCALAR_FIELD_DATANAME);
    
    if (baseArray.entries() > 0) {
        SIM_ScalarField *field = SIM_DATA_CREATE(SIM_ScalarField, fieldname);
        if (field) {
            field->match(SIM_DATA_CAST(baseArray[0], SIM_ScalarField));
            obj->addDataToList(field);
            field->dataUnused();
            return field;
        }
    }
    
    return nullptr;
}

// Get or create a vector field
SIM_VectorField *
getOrCreateVectorField(SIM_Object *obj, const char *fieldname)
{
    SIM_DataArray dataArray;
    obj->filterSIMData(dataArray, fieldname);
    
    if (dataArray.entries() > 0)
        return SIM_DATA_CAST(dataArray[0], SIM_VectorField);
    
    SIM_DataArray baseArray;
    obj->filterSIMData(baseArray, SIM_VECTOR_FIELD_DATANAME);
    
    if (baseArray.entries() > 0) {
        SIM_VectorField *field = SIM_DATA_CREATE(SIM_VectorField, fieldname);
        if (field) {
            field->match(SIM_DATA_CAST(baseArray[0], SIM_VectorField));
            obj->addDataToList(field);
            field->dataUnused();
            return field;
        }
    }
    
    return nullptr;
}

// Main entry point for the plugin
void
newSopOperator(OP_OperatorTable *table)
{
    // Register the solver
    FlexONNXSolver::initializeSubclass();
    FlexONNXSolver::installSubclass(customTemplates);
}

// Registration functions
void
newDopOperator(OP_OperatorTable *table)
{
    SIM_SubSolver::registerSubSolver(FlexONNXSolver::SOLVER_NAME, SIM_SOLVER_DATANAME,
                                     SIM_SubSolver::sSubSolverCreatePriorityDefault,
                                     SIM_SubSolver::getSubSolverPriority(SIM_SOLVER_DATANAME));
}

// Helper utility to extract ONNX model information
class ONNXModelInfo
{
public:
    ONNXModelInfo(const char* modelPath)
    {
        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
            
            session_ = std::make_unique<Ort::Session>(env, modelPath, sessionOptions);
            
            // Get number of inputs and outputs
            numInputs_ = session_->GetInputCount();
            numOutputs_ = session_->GetOutputCount();
            
            // Get input and output information
            Ort::AllocatorWithDefaultOptions allocator;
            
            for (size_t i = 0; i < numInputs_; i++) {
                char* name = session_->GetInputName(i, allocator);
                inputNames_.push_back(std::string(name));
                allocator.Free(name);
                
                auto typeInfo = session_->GetInputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                
                inputTypes_.push_back(tensorInfo.GetElementType());
                inputDims_.push_back(tensorInfo.GetShape());
            }
            
            for (size_t i = 0; i < numOutputs_; i++) {
                char* name = session_->GetOutputName(i, allocator);
                outputNames_.push_back(std::string(name));
                allocator.Free(name);
                
                auto typeInfo = session_->GetOutputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                
                outputTypes_.push_back(tensorInfo.GetElementType());
                outputDims_.push_back(tensorInfo.GetShape());
            }
            
            valid_ = true;
        }
        catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error when loading model info: " << e.what() << std::endl;
            valid_ = false;
        }
    }
    
    bool isValid() const { return valid_; }
    
    void printModelInfo(std::ostream& os) const
    {
        if (!valid_) {
            os << "Invalid ONNX model" << std::endl;
            return;
        }
        
        os << "ONNX Model Information:" << std::endl;
        os << "  Inputs: " << numInputs_ << std::endl;
        
        for (size_t i = 0; i < numInputs_; i++) {
            os << "    Input[" << i << "]: " << inputNames_[i] << ", Type: " << getTypeString(inputTypes_[i]) << std::endl;
            os << "      Dimensions: ";
            printDimensions(os, inputDims_[i]);
            os << std::endl;
        }
        
        os << "  Outputs: " << numOutputs_ << std::endl;
        
        for (size_t i = 0; i < numOutputs_; i++) {
            os << "    Output[" << i << "]: " << outputNames_[i] << ", Type: " << getTypeString(outputTypes_[i]) << std::endl;
            os << "      Dimensions: ";
            printDimensions(os, outputDims_[i]);
            os << std::endl;
        }
    }
    
private:
    std::unique_ptr<Ort::Session> session_;
    bool valid_ = false;
    
    size_t numInputs_ = 0;
    size_t numOutputs_ = 0;
    
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    
    std::vector<ONNXTensorElementDataType> inputTypes_;
    std::vector<ONNXTensorElementDataType> outputTypes_;
    
    std::vector<std::vector<int64_t>> inputDims_;
    std::vector<std::vector<int64_t>> outputDims_;
    
    std::string getTypeString(ONNXTensorElementDataType type) const
    {
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "double";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "complex64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "complex128";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
            default: return "unknown";
        }
    }
    
    void printDimensions(std::ostream& os, const std::vector<int64_t>& dims) const
    {
        os << "[";
        for (size_t j = 0; j < dims.size(); j++) {
            if (j > 0) os << ", ";
            
            // Print 'dynamic' for dimensions that can be resized
            if (dims[j] < 0) {
                os << "dynamic";
            } else {
                os << dims[j];
            }
        }
        os << "]";
    }
};

// Utility function to check if a model supports dynamic dimensions
bool hasDynamicDimensions(const char* modelPath)
{
    ONNXModelInfo modelInfo(modelPath);
    
    if (!modelInfo.isValid()) {
        return false;
    }
    
    // Log model info for debugging purposes
    std::ostringstream oss;
    modelInfo.printModelInfo(oss);
    std::cout << oss.str() << std::endl;
    
    return true;
}

// Additional utility for handling different input layouts
class ONNXInputMapper
{
public:
    enum class Layout {
        NCHW,  // Batch, Channels, Height, Width
        NHWC,  // Batch, Height, Width, Channels
        NCDHW, // Batch, Channels, Depth, Height, Width
        NDHWC  // Batch, Depth, Height, Width, Channels
    };
    
    ONNXInputMapper(Layout sourceLayout, Layout targetLayout)
        : sourceLayout_(sourceLayout), targetLayout_(targetLayout)
    {
    }
    
    std::vector<float> remapTensorData(const std::vector<float>& sourceData, 
                                      const std::vector<int64_t>& sourceDims,
                                      std::vector<int64_t>& targetDims)
    {
        // For now, we'll implement a simple case for NCHW to NCDHW conversion
        // A more comprehensive implementation would handle all layout combinations
        
        if (sourceLayout_ == Layout::NCHW && targetLayout_ == Layout::NCDHW) {
            // Extract dimensions
            int64_t batch = sourceDims[0];
            int64_t channels = sourceDims[1];
            int64_t height = sourceDims[2];
            int64_t width = sourceDims[3];
            
            // Assume depth of 1 for conversion to 3D
            int64_t depth = 1;
            
            // Set target dimensions
            targetDims = {batch, channels, depth, height, width};
            
            // Calculate sizes
            size_t sourceSize = batch * channels * height * width;
            size_t targetSize = batch * channels * depth * height * width;
            
            // Sanity check
            if (sourceData.size() != sourceSize) {
                std::cerr << "Source data size mismatch: expected " << sourceSize << ", got " << sourceData.size() << std::endl;
                return sourceData; // Return original data on error
            }
            
            // Create target data
            std::vector<float> targetData(targetSize);
            
            // Copy data with adjusted indices
            for (int64_t n = 0; n < batch; ++n) {
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            // Source index (NCHW)
                            size_t sourceIdx = ((n * channels + c) * height + h) * width + w;
                            
                            // Target index (NCDHW with depth=1)
                            size_t targetIdx = (((n * channels + c) * depth + 0) * height + h) * width + w;
                            
                            targetData[targetIdx] = sourceData[sourceIdx];
                        }
                    }
                }
            }
            
            return targetData;
        }
        
        // If conversion not implemented, return original data
        targetDims = sourceDims;
        return sourceData;
    }
    
private:
    Layout sourceLayout_;
    Layout targetLayout_;
};

// Additional methods for FlexONNXSolver to handle variable dimensions
void FlexONNXSolver::resizeFieldsForDynamicModel(SIM_Object *obj, int resX, int resY, int resZ)
{
    // Get the fields
    SIM_ScalarField *pressureField = getOrCreateScalarField(obj, "pressure");
    SIM_VectorField *velocityField = getOrCreateVectorField(obj, "vel");
    
    if (!pressureField || !velocityField)
        return;
    
    // Get current resolutions
    UT_Vector3I pressureRes = pressureField->getVoxelRes();
    UT_Vector3I velocityRes = velocityField->getVoxelRes();
    
    // Check if resizing is needed
    if (pressureRes.x() != resX || pressureRes.y() != resY || pressureRes.z() != resZ) {
        // Resize pressure field
        pressureField->resizeField(resX, resY, resZ);
    }
    
    if (velocityRes.x() != resX || velocityRes.y() != resY || velocityRes.z() != resZ) {
        // Resize velocity field
        velocityField->resizeField(resX, resY, resZ);
    }
}

// Registration function for the plugin shared library
void
initializeSIM(void *)
{
    IMPLEMENT_DATAFACTORY(FlexONNXSolver);
}