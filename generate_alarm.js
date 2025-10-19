// 生成噪声警报音频文件
function generateNoiseAudio() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const sampleRate = audioContext.sampleRate;
    const duration = 3; // 3秒
    const frameCount = sampleRate * duration;
    
    const audioBuffer = audioContext.createBuffer(1, frameCount, sampleRate);
    const output = audioBuffer.getChannelData(0);
    
    // 生成白噪声
    for (let i = 0; i < frameCount; i++) {
        output[i] = Math.random() * 2 - 1; // 随机值在-1到1之间
    }
    
    // 创建音频源
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    
    // 添加增益控制
    const gainNode = audioContext.createGain();
    gainNode.gain.value = 0.3; // 降低音量
    
    source.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    return { source, audioContext };
}

// 导出函数
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { generateNoiseAudio };
}