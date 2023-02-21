#version 420
precision highp float;
precision highp int32_t;

/* NOTE
頂点バッファを使わず全画面に描画する頂点シェーダ
https://shobomaru.wordpress.com/2014/12/31/shader-only-fullscreen-quad/
*/

layout(location = 0) out vec2 uv;

void main() {
    float x = (gl_VertexID & 1) != 0 ? 1.0 : 0.0;
    float y = (gl_VertexID & 2) != 0 ? 1.0 : 0.0;
    gl_Position = vec4(vec2(x, y) * 2.0 - 1.0, 0, 1);
    uv = vec2(x, y);
}
