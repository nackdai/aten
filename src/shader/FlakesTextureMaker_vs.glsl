#version 450
precision highp float;
precision highp int;

/* NOTE
頂点バッファを使わず全画面に描画する頂点シェーダ
https://shobomaru.wordpress.com/2014/12/31/shader-only-fullscreen-quad/
*/

void main()
{
    int x = (gl_VertexID & 1) * 2 - 1;
    int y = (gl_VertexID & 2) * 2 - 1;
    gl_Position = vec4(x, y, 0, 1);
}
