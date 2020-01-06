#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../../../01_PatternExtractionRecognition/WhitePatternDetectionLib/ComponentMarker.h"

#include <iostream>

namespace py = pybind11;

cv::Mat npUint8_1cToCVMat(py::array_t<unsigned char>& input) {
	if (input.ndim() != 2) {
		throw std::runtime_error("1-channel image must be 2 dims");
	}

	py::buffer_info buf = input.request();
	cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);

	return mat;
}

py::list componentsExtractor(py::array_t<unsigned char>& input, int color = 255, int maxSize = 100000) {
	auto img = npUint8_1cToCVMat(input);
	std::cout 
		<< "Img rows: " << img.rows << " cols: " << img.cols 
		<< "\nThe first pixel value is: " << (int)img.at<unsigned char>(0, 0) << "\n";
	
	ComponentMarker<uint8_t> cm;
	cm.setInputBinaryImage(img);
	if (color ==  255)
	{
		cm.setComponentColor(ComponentMarker<uint8_t>::White);
	}
	else if (color == 0)
	{
		cm.setComponentColor(ComponentMarker<uint8_t>::Black);
	}
	
	cm.markComponent();
	std::cout << "Finished marking components.\n";
	DepthComponents & components = cm.getDepthComponents();
	py::list output;
	for (auto & component : components)
	{
		if (component.size() > maxSize)
		{
			continue;
		}
		py::list l;
		//py::list l = py::cast(component);
		//std::cout << "Component size: " << component.size() << "\n";
		for (int j = 0; j < component.size(); ++j) {
			py::tuple c(2);
			c[0] = (int)component[j].x;
			c[1] = (int)component[j].y;
			l.append(c);
		}
		output.append(l);
	}
	return output;
}

PYBIND11_MODULE(ComponentExtractor, m) {
	m.doc() = R"pbdoc(
        Test img process
        -----------------------

        .. currentmodule:: WhitePatternsExtractionPyModule

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

	m.def("componentsExtractor", &componentsExtractor, R"pbdoc(
	    Print a set of strings.
	
	)pbdoc", py::arg("input image"), py::arg("color") = 255, py::arg("maxSize") = 100000);


#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}