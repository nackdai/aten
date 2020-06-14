#pragma once

#include "aten.h"
#include "idaten.h"

class MaterialEditWindow {
private:
    MaterialEditWindow();
    ~MaterialEditWindow();

public:
    static bool init(
        int width, int height,
        const char* titile);

    static void notifyPickMtrlId(int mtrlid);

    static void buildScene();

    using FuncPickMtrlIdNotifier = std::function<void(int)>;

    static void setFuncChangeMtrlIdNotifier(FuncPickMtrlIdNotifier func)
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
    static aten::window* s_wnd;

    static aten::PinholeCamera s_camera;
    static bool s_isCameraDirty;

    static aten::AcceleratedScene<aten::GPUBvh> s_scene;

    static idaten::PathTracing s_tracer;

    static aten::GammaCorrection s_gamma;
    static aten::visualizer* s_visualizer;

    static bool s_willShowGUI;
    static bool s_willTakeScreenShot;
    static int s_cntScreenShot;

    static int s_maxSamples;
    static int s_maxBounce;

    static bool s_isMouseLBtnDown;
    static bool s_isMouseRBtnDown;
    static int s_prevX;
    static int s_prevY;

    static int s_width;
    static int s_height;

    static int s_pickedMtrlId;
    static bool s_needUpdateMtrl;
    static std::vector<aten::material*> s_mtrls;
    static std::vector<const char*> s_mtrlNames;

    static FuncPickMtrlIdNotifier s_pickMtrlIdNotifier;
};
