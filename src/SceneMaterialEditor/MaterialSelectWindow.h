#pragma once

#include "aten.h"

class MaterialSelectWindow {
private:
	MaterialSelectWindow();
	~MaterialSelectWindow();

public:
	static bool init(
		int width, int height,
		const char* titile,
		const char* input);

	using FuncPickMtrlIdNotifier = std::function<void(int)>;

	static void setFuncPickMtrlIdNotifier(FuncPickMtrlIdNotifier func)
	{
		s_pickMtrlIdNotifier = func;
	}

private:
	static void onRun(aten::window* window);
	static void onClose();
	static void onMouseBtn(bool left, bool press, int x, int y);
	static void onMouseMove(int x, int y);
	static void onMouseWheel(int delta);
	static void onKey(bool press, aten::Key key);

private:
	static aten::PinholeCamera s_camera;
	static bool s_isCameraDirty;

	static aten::object* s_obj;
	static aten::RasterizeRenderer s_rasterizer;
	
	static aten::Blitter s_blitter;
	static aten::visualizer* s_visualizer;

	static bool s_willTakeScreenShot;
	static int s_cntScreenShot;

	static bool s_isMouseLBtnDown;
	static bool s_isMouseRBtnDown;
	static int s_prevX;
	static int s_prevY;

	static int s_width;
	static int s_height;

	static bool s_willPick;
	static bool s_pick;

	static aten::FBO s_fbo;

	static std::vector<aten::TColor<uint8_t, 4>> s_attrib;

	static int s_pickedMtrlId;

	static FuncPickMtrlIdNotifier s_pickMtrlIdNotifier;
};