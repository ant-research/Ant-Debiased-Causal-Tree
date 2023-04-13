#include "include/gbct_utils.h"
#include "include/agg_fn.h"
#include "include/did_utils.h"
#include "include/gbct_bin.h"
#include "include/bin.h"
#include "include/predict.h"
#include "src/dense_bin.hpp"


using namespace gbct_utils;

PYBIND11_MODULE(gbct_utils, m)
{
    m.doc() = "pybind11 for gbct"; // optional module docstring

    {        
        auto m_comm = m.def_submodule("common", "gbct common utils");

        m_comm.def("sumby_double_int32", &sumby_nogil<double, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_float_int32", &sumby_nogil<float, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_float_int64", &sumby_nogil<float, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_int32_int32", &sumby_nogil<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_int64_int64", &sumby_nogil<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_int32_int64", &sumby_nogil<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_int64_int32", &sumby_nogil<int64_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_float_uint32", &sumby_nogil<float, uint32_t>, py::return_value_policy::move);
        
        m_comm.def("sumby_parallel_double_int32", &sumby_parallel<double, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_float_int32", &sumby_parallel<float, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_float_int64", &sumby_parallel<float, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int32_int32", &sumby_parallel<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int64_int64", &sumby_parallel<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int32_int64", &sumby_parallel<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int64_int32", &sumby_parallel<int64_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_float_uint32", &sumby_parallel<float, uint32_t>, py::return_value_policy::move);

        m_comm.def("countby_double_uint32", &countby_nogil<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_double_int32", &countby_nogil<double, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_double_int64", &countby_nogil<double, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_double_uint64", &countby_nogil<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_float_int64", &countby_nogil<float, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_float_uint64", &countby_nogil<float, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_float_int32", &countby_nogil<float, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_float_uint32", &countby_nogil<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_int32_int32", &countby_nogil<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_int64_int64", &countby_nogil<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_int32_int64", &countby_nogil<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_int64_int32", &countby_nogil<int64_t, int32_t>, py::return_value_policy::move);


        m_comm.def("countby_parallel_double_uint32", &countby_parallel<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_double_int32", &countby_parallel<double, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_double_int64", &countby_parallel<double, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_double_uint64", &countby_parallel<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_int64", &countby_parallel<float, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_uint64", &countby_parallel<float, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_int32", &countby_parallel<float, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_uint32", &countby_parallel<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int32_int32", &countby_parallel<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int64_int64", &countby_parallel<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int32_int64", &countby_parallel<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int64_int32", &countby_parallel<int64_t, int32_t>, py::return_value_policy::move);

        m_comm.def("indexbyarray_double_int32", &indexbyarray<double, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_double_uint32", &indexbyarray<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_double_int64", &indexbyarray<double, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_double_uint64", &indexbyarray<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_int32", &indexbyarray<float, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_uint32", &indexbyarray<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_int64", &indexbyarray<float, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_uint64", &indexbyarray<float, uint64_t>, py::return_value_policy::move);

        m_comm.def("indexbyarray2_double_int32", &indexbyarray2<double, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_double_uint32", &indexbyarray2<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_double_int64", &indexbyarray2<double, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_double_uint64", &indexbyarray2<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_int32", &indexbyarray2<float, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_uint32", &indexbyarray2<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_int64", &indexbyarray2<float, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_uint64", &indexbyarray2<float, uint64_t>, py::return_value_policy::move);

        m_comm.def("update_x_map_int32", &update_x_map<int32_t>, py::return_value_policy::move);
        m_comm.def("update_histogram_double_int32", &update_histogram<double, int32_t>, py::return_value_policy::move);
        m_comm.def("update_histogram_int32_int32", &update_histogram<int32_t, int32_t>, py::return_value_policy::move);

        py::class_<Node>(m_comm, "CppNode")
            // .def(py::init<>())
            .def_readwrite("split_feature", &Node::split_feature)
            .def_readwrite("split_thresh", &Node::split_thresh)
            .def_property("children", &Node::get_children, &Node::set_children)
            .def_readwrite("is_leaf", &Node::is_leaf)
            .def_readwrite("leaf_id", &Node::leaf_id)
            .def_readwrite("level_id", &Node::level_id)
            .def("get_property_keys", &Node::get_property_keys);

        py::class_<DebiasNode, Node>(m_comm, "CppDebiasNode")
            .def(py::init<>())
            // .def_readwrite("outcomes", &DebiasNode::outcomes)
            .def_property("outcomes", &DebiasNode::get_outcomes, &DebiasNode::set_outcomes)
            .def_readwrite("bias", &DebiasNode::bias)
            .def_readwrite("eta", &DebiasNode::eta)
            .def_readwrite("debiased_effect", &DebiasNode::debiased_effect)
            .def_readwrite("effect", &DebiasNode::effect)
            .def("get_property_keys", &DebiasNode::get_property_keys)
            .def("get_info", &DebiasNode::get_info)
            .def("set_info", &DebiasNode::set_info);

        py::class_<DiDNode, DebiasNode>(m_comm, "CppDiDNode")
            .def("get_info", &DiDNode::get_info)
            .def("set_info", &DiDNode::set_info)
            .def(py::init<>())
            .def("get_property_keys", &DiDNode::get_property_keys);

        m_comm.def("predict_debias", &predict<DebiasNode>, py::return_value_policy::move);
        m_comm.def("predict_did", &predict<DiDNode>, py::return_value_policy::move);

        m_comm.def("calculate_loss", &calculate_loss<double, int32_t>, py::return_value_policy::move);
        m_comm.def("sum", py::overload_cast<const py::array_t<double> &, int, bool >(&sum<double>));
        m_comm.def("concatenate", &concatenate<double>);
        // just for unittest
        m_comm.def("array_add", [](const py::array_t<double> & a, const py::array_t<double> & b){return a+b;});
    }

    {
        auto m_bin = m.def_submodule("bin", "gbct bin");
        m_bin.def("FindBinParallel_double", &FindBinParallel<double>);
        m_bin.def("Value2BinParallel_double_int32", &Value2BinParallel<double, int32_t>);
        m_bin.def("Value2BinParallel_double_uint32", &Value2BinParallel<double, uint32_t>);
        m_bin.def("Value2BinParallel_float_int32", &Value2BinParallel<float, int32_t>);
        m_bin.def("Value2BinParallel_float_uint32", &Value2BinParallel<float, uint32_t>);

        py::class_<LightGBM::BinMapper>(m_bin, "BinMaper")
            .def(py::init<>())
            .def("GetUpperBoundValue", &LightGBM::BinMapper::GetUpperBoundValue)
            .def("is_trivial", &LightGBM::BinMapper::is_trivial)
            .def("BinToValue", &LightGBM::BinMapper::BinToValue)
            .def("MaxCatValue", &LightGBM::BinMapper::MaxCatValue)
            .def("SizesInByte", &LightGBM::BinMapper::SizesInByte)
            .def("ValueToBin", &LightGBM::BinMapper::ValueToBin)
            .def("GetDefaultBin", &LightGBM::BinMapper::GetDefaultBin)
            .def("GetMostFreqBin", &LightGBM::BinMapper::GetMostFreqBin)
            .def("bin_type", &LightGBM::BinMapper::bin_type)
            .def("bin_info_string", &LightGBM::BinMapper::bin_info_string)
            .def("sparse_rate", &LightGBM::BinMapper::sparse_rate)
            .def(py::pickle([](const LightGBM::BinMapper& b){
                return py::make_tuple(b.num_bin(), b.missing_type(), b.GetUpperBoundValue(), b.is_trivial(), b.sparse_rate()
                    , b.bin_type(), b.categorical_2_bin(), b.bin_2_categorical(), b.min_val(), b.max_val()
                    , b.GetDefaultBin(), b.GetMostFreqBin());
            }, [](py::tuple t){// __setstate_
                auto b = LightGBM::BinMapper(t[0].cast<int>(), t[1].cast<LightGBM::MissingType>(), 
                    t[2].cast<std::vector<double>>(), t[3].cast<bool>(), t[4].cast<double>(), 
                    t[5].cast<LightGBM::BinType>(), t[6].cast<const std::unordered_map<int, unsigned int>>(), 
                    t[7].cast<const std::vector<int>>(), t[8].cast<double>(), t[9].cast<double>(), t[10].cast<uint32_t>(), 
                    t[11].cast<uint32_t>());
                return b;
            }));

        py::class_<LightGBM::Bin>(m_bin, "Bin")
        .def(py::init(&LightGBM::Bin::CreateDenseBin))
        .def(py::init(&LightGBM::Bin::CreateSparseBin))
        .def("Push", &LightGBM::Bin::Push);

        py::enum_<LightGBM::MissingType>(m_bin, "MissingType")
            .value("None", LightGBM::MissingType::None)
            .value("Zero", LightGBM::MissingType::Zero)
            .value("NaN", LightGBM::MissingType::NaN)
            .export_values();

        py::enum_<LightGBM::BinType>(m_bin, "BinType")
            .value("NumericalBin", LightGBM::BinType::NumericalBin)
            .value("CategoricalBin", LightGBM::BinType::CategoricalBin)
            .export_values();
    }

}
