// 无人机检测系统前端控制
class DroneDetectionSystem {
    constructor() {
        this.socket = null;
        this.isDetecting = false;
        this.cameraStream = null;
        this.detectionInterval = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.videoCanvas = null;
        this.videoCtx = null;
        this.countdownInterval = null;
        this.currentSession = null;
        
        // 检测统计
        this.stats = {
            totalDetections: 0,
            consecutiveDetections: 0,
            lastDetectionTime: null,
            droneCount: 0,
            avgConfidence: 0,
            confidenceSum: 0,
            totalDroneCount: 0
        };
        
        // 检测历史记录
        this.detectionHistory = [];
        
        // 用户配置
        this.config = {
            detectionInterval: 10000, // 10秒检测间隔
            alertThreshold: 2, // 连续检测2次触发警报
            multiDroneThreshold: 2, // 2个或以上无人机触发高级警报
            confidenceThreshold: 0.5,
            soundAlerts: true,
            desktopNotifications: true
        };
        
        console.log('🔧 初始配置:', this.config);
        
        // 警报音频系统
        this.audioContext = null;
        this.currentAlarmAudio = null;
        this.isAlarmPlaying = false;
        this.previousDroneCount = 0; // 跟踪之前的无人机数量
        this.initAudio();
        
        // 立即加载保存的配置
        this.loadSavedData();
        
        // 添加音频测试按钮
        this.addAudioTestButton();
        
        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupCanvas();
        this.updateUI();
    }

    setupWebSocket() {
        this.socket = io('http://localhost:3000');
        
        this.socket.on('connect', () => {
            console.log('WebSocket连接成功');
            this.updateSystemStatus('在线');
        });
        
        this.socket.on('disconnect', () => {
            console.log('WebSocket连接断开');
            this.updateSystemStatus('离线');
        });
        
        this.socket.on('system-status', (data) => {
            // 保持音频配置不被服务器覆盖
            const soundAlerts = this.config.soundAlerts;
            const desktopNotifications = this.config.desktopNotifications;
            
            this.config = { ...data.config, soundAlerts, desktopNotifications };
            this.stats = data.stats;
            
            console.log('📡 收到服务器配置更新:', {
                serverConfig: data.config,
                mergedConfig: this.config
            });
            
            this.updateUI();
        });
        
        this.socket.on('detection-result', (result) => {
            this.handleDetectionResult(result);
        });
        
        this.socket.on('detection-error', (error) => {
            console.error('检测错误:', error);
            this.showAlert('检测错误: ' + error.error, 'error');
        });
        
        this.socket.on('camera-detection-started', () => {
            console.log('摄像头检测已启动');
        });
        
        this.socket.on('camera-detection-stopped', () => {
            console.log('摄像头检测已停止');
        });
    }

    setupCanvas() {
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // 获取video元素
        this.video = document.getElementById('videoElement');
        
        // 设置视频检测canvas
        this.videoCanvas = document.getElementById('videoCanvas');
        if (this.videoCanvas) {
            this.videoCtx = this.videoCanvas.getContext('2d');
        }
    }

    initAudio() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log('音频上下文初始化成功');
        } catch (error) {
            console.warn('Web Audio API不支持:', error);
        }
    }

    async playNoiseAlarm() {
        try {
            console.log('🔊 开始播放警报音频...');
            console.log('当前配置 - soundAlerts:', this.config.soundAlerts);
            console.log('浏览器信息:', {
                userAgent: navigator.userAgent,
                audioContextSupport: !!(window.AudioContext || window.webkitAudioContext),
                webkitAudioContext: !!window.webkitAudioContext,
                audioContext: !!window.AudioContext
            });
            
            // 检查音频权限
            const hasPermission = await this.checkAudioPermissions();
            if (!hasPermission) {
                console.log('❌ 音频权限检查失败，无法播放警报');
                return;
            }
            
            // 如果已经在播放警报，先停止
            if (this.isAlarmPlaying) {
                console.log('⏹️ 停止当前播放的警报');
                this.stopAlarm();
            }
            
            // 确保AudioContext已初始化
            if (!this.audioContext) {
                console.log('🔧 初始化音频上下文...');
                await this.initAudioContext();
            }
            
            // 如果音频上下文被暂停，尝试恢复
            if (this.audioContext && this.audioContext.state === 'suspended') {
                console.log('▶️ 恢复暂停的音频上下文...');
                await this.audioContext.resume();
                console.log('音频上下文状态:', this.audioContext.state);
            }
            
            if (!this.audioContext) {
                console.error('❌ 音频上下文初始化失败');
                this.showAlert('音频系统不可用', 'error');
                return;
            }
            
            console.log('✅ 音频上下文状态:', this.audioContext.state);
            console.log('🎛️ 音频上下文详细信息:', {
                sampleRate: this.audioContext.sampleRate,
                currentTime: this.audioContext.currentTime,
                destination: this.audioContext.destination,
                state: this.audioContext.state
            });
            
            // 生成噪声音频
            const sampleRate = this.audioContext.sampleRate;
            const duration = 3; // 3秒
            const frameCount = sampleRate * duration;
            
            console.log('🎵 生成音频缓冲区:', {
                sampleRate: sampleRate,
                duration: duration,
                frameCount: frameCount
            });
            
            const audioBuffer = this.audioContext.createBuffer(1, frameCount, sampleRate);
            const output = audioBuffer.getChannelData(0);
            
            // 生成白噪声
            for (let i = 0; i < frameCount; i++) {
                output[i] = (Math.random() * 2 - 1) * 0.8; // 增加音量到0.8
            }
            
            console.log('🔊 白噪声生成完成，样本数:', frameCount);
            
            // 创建音频源
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.loop = true; // 循环播放
            
            // 添加增益控制
            const gainNode = this.audioContext.createGain();
            gainNode.gain.value = 0.9; // 增加音量到0.9
            
            console.log('🎚️ 音频节点连接:', {
                source: source,
                gainValue: gainNode.gain.value,
                loop: source.loop
            });
            
            source.connect(gainNode);
            gainNode.connect(this.audioContext.destination);
            
            // 添加事件监听器
            source.onended = () => {
                console.log('🔇 音频播放结束');
                this.isAlarmPlaying = false;
                this.hideStopAlarmButton();
            };
            
            // 开始播放
            console.log('▶️ 开始播放音频...');
            source.start();
            
            this.currentAlarmAudio = source;
            this.isAlarmPlaying = true;
            
            // 显示停止按钮
            this.showStopAlarmButton();
            
            console.log('🎵 噪声警报音频播放中...');
            console.log('音频参数: 采样率=' + sampleRate + ', 时长=' + duration + '秒, 音量=' + gainNode.gain.value);
            this.showAlert('🔊 警报音频正在播放', 'warning');
            
        } catch (error) {
            console.error('❌ 播放噪声警报失败:', error);
            console.error('错误详情:', {
                name: error.name,
                message: error.message,
                stack: error.stack
            });
            
            // 尝试重新初始化AudioContext
            try {
                console.log('🔄 尝试重新初始化AudioContext...');
                await this.initAudioContext();
                console.log('✅ AudioContext重新初始化成功');
            } catch (initError) {
                console.error('❌ AudioContext重新初始化失败:', initError);
            }
            
            this.showAlert('播放警报音频失败: ' + error.message, 'error');
        }
    }
    
    stopAlarm() {
        try {
            if (this.currentAlarmAudio) {
                this.currentAlarmAudio.stop();
                this.currentAlarmAudio.disconnect();
                this.currentAlarmAudio = null;
            }
            this.isAlarmPlaying = false;
            this.hideStopAlarmButton();
            console.log('警报音频已停止');
        } catch (error) {
            console.error('停止警报音频失败:', error);
        }
    }
    
    showStopAlarmButton() {
        let stopButton = document.getElementById('stopAlarmButton');
        if (!stopButton) {
            stopButton = document.createElement('button');
            stopButton.id = 'stopAlarmButton';
            stopButton.innerHTML = '<i class="fas fa-stop"></i> 停止警报';
            stopButton.className = 'fixed bottom-4 right-4 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-full shadow-lg z-50 flex items-center gap-2';
            stopButton.onclick = () => this.stopAlarm();
            document.body.appendChild(stopButton);
        }
        stopButton.style.display = 'flex';
    }
    
    hideStopAlarmButton() {
        const stopButton = document.getElementById('stopAlarmButton');
        if (stopButton) {
            stopButton.style.display = 'none';
        }
    }

    async initAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log('音频上下文初始化成功，状态:', this.audioContext.state);
        } catch (error) {
            console.warn('Web Audio API不支持:', error);
        }
    }

    async activateAudioContext() {
        try {
            if (!this.audioContext) {
                await this.initAudioContext();
            }
            
            if (this.audioContext && this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
                console.log('音频上下文已激活，状态:', this.audioContext.state);
                this.showAlert('音频系统已激活', 'success');
            } else if (this.audioContext) {
                console.log('音频上下文当前状态:', this.audioContext.state);
            }
        } catch (error) {
            console.error('激活音频上下文失败:', error);
            this.showAlert('音频系统激活失败: ' + error.message, 'warning');
        }
    }
    
    async checkAudioPermissions() {
        try {
            // 检查浏览器是否支持音频播放
            if (!window.AudioContext && !window.webkitAudioContext) {
                console.warn('浏览器不支持Web Audio API');
                this.showAlert('⚠️ 您的浏览器不支持音频播放功能', 'warning');
                return false;
            }
            
            // 检查音频上下文状态
            if (this.audioContext && this.audioContext.state === 'running') {
                console.log('音频权限检查通过');
                return true;
            }
            
            // 提示用户需要交互来激活音频
            const userConfirm = confirm('🔊 为了播放警报声音，需要您的确认。点击"确定"激活音频功能。');
            if (userConfirm) {
                await this.activateAudioContext();
                return this.audioContext && this.audioContext.state === 'running';
            } else {
                this.showAlert('⚠️ 音频功能未激活，将无法播放警报声音', 'warning');
                return false;
            }
        } catch (error) {
            console.error('音频权限检查失败:', error);
            this.showAlert('❌ 音频权限检查失败: ' + error.message, 'error');
            return false;
        }
    }
    
    addAudioTestButton() {
        // 创建音频测试按钮
        const testButton = document.createElement('button');
        testButton.id = 'audioTestButton';
        testButton.innerHTML = '<i class="fas fa-volume-up"></i> 测试音频';
        testButton.className = 'fixed top-4 right-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-full shadow-lg z-50 flex items-center gap-2';
        testButton.onclick = () => this.testAudio();
        document.body.appendChild(testButton);
        
        // 创建强制音频播放按钮
        const forceButton = document.createElement('button');
        forceButton.id = 'forceAudioButton';
        forceButton.innerHTML = '<i class="fas fa-play"></i> 强制播放';
        forceButton.className = 'fixed top-16 right-4 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-full shadow-lg z-50 flex items-center gap-2';
        forceButton.onclick = () => this.forcePlayAudio();
        document.body.appendChild(forceButton);
        
        console.log('✅ 音频测试按钮已添加');
    }
    
    async testAudio() {
        console.log('🧪 开始音频测试...');
        console.log('🔍 当前系统状态:', {
            soundAlerts: this.config.soundAlerts,
            isAlarmPlaying: this.isAlarmPlaying,
            audioContext: this.audioContext ? {
                state: this.audioContext.state,
                sampleRate: this.audioContext.sampleRate,
                currentTime: this.audioContext.currentTime
            } : null,
            browserSupport: {
                AudioContext: !!window.AudioContext,
                webkitAudioContext: !!window.webkitAudioContext
            }
        });
        
        try {
            // 强制启用音频警报进行测试
            const originalSoundAlerts = this.config.soundAlerts;
            this.config.soundAlerts = true;
            
            this.showAlert('🧪 正在测试音频播放...', 'info');
            
            // 模拟无人机数量变化来触发音频
            console.log('🎯 模拟无人机检测，触发音频播放');
            this.previousDroneCount = 0;
            this.updateDroneCount(1);
            
            // 3秒后停止测试
            setTimeout(() => {
                console.log('⏹️ 停止音频测试');
                this.updateDroneCount(0);
                this.config.soundAlerts = originalSoundAlerts;
                this.showAlert('🧪 音频测试完成', 'success');
            }, 3000);
            
        } catch (error) {
            console.error('❌ 音频测试失败:', error);
            this.showAlert('音频测试失败: ' + error.message, 'error');
        }
    }
    
    async forcePlayAudio() {
        console.log('🎵 强制播放音频测试开始');
        
        try {
            // 强制激活音频上下文
            await this.activateAudioContext();
            
            console.log('🔧 音频上下文状态:', this.audioContext?.state);
            
            // 直接调用播放函数，绕过所有条件检查
            await this.playNoiseAlarm();
            
            this.showAlert('🎵 强制音频播放中...', 'info');
            console.log('✅ 强制音频播放成功启动');
            
            // 3秒后停止
            setTimeout(() => {
                this.stopAlarm();
                this.showAlert('🔇 强制音频播放结束', 'success');
                console.log('🔇 强制音频播放结束');
            }, 3000);
            
        } catch (error) {
            console.error('❌ 强制音频播放失败:', error);
            this.showAlert('强制音频播放失败: ' + error.message, 'error');
        }
    }

    setupEventListeners() {
        // 摄像头控制按钮
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        if (startBtn) startBtn.addEventListener('click', () => this.startCamera());
        if (stopBtn) stopBtn.addEventListener('click', () => this.stopCamera());
        
        // 图片上传
        const uploadInput = document.getElementById('imageUpload');
        if (uploadInput) uploadInput.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // 系统配置
        const configBtn = document.getElementById('configBtn');
        if (configBtn) configBtn.addEventListener('click', () => this.showConfigModal());
        
        // 检测历史
        const historyBtn = document.getElementById('historyBtn');
        if (historyBtn) historyBtn.addEventListener('click', () => this.showHistoryModal());
        
        // 模态框关闭
        document.querySelectorAll('.close-modal').forEach(btn => {
            btn.addEventListener('click', (e) => this.closeModal(e.target.closest('.modal')));
        });
        
        // 配置保存
        const saveConfigBtn = document.getElementById('saveConfig');
        if (saveConfigBtn) saveConfigBtn.addEventListener('click', () => this.saveConfig());
        
        // 清除历史
        const clearHistoryBtn = document.getElementById('clearHistory');
        if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', () => this.clearHistory());
    }

    async startCamera() {
        try {
            // 激活音频上下文（需要用户交互）
            await this.activateAudioContext();
            
            this.showLoading(true);
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            });
            
            this.video.srcObject = stream;
            this.cameraStream = stream;
            
            // 等待视频加载
            await new Promise((resolve) => {
                this.video.onloadedmetadata = resolve;
            });
            
            // 创建新的检测会话
            this.currentSession = {
                timestamp: new Date(),
                droneCount: 0,
                avgConfidence: 0,
                detections: []
            };
            
            this.isDetecting = true;
            this.updateCameraButtons();
            this.updateSystemStatus('运行中');
            
            // 开始检测循环
            this.startDetectionLoop();
            
            console.log('摄像头启动成功');
            this.showAlert('摄像头启动成功', 'success');
            
        } catch (error) {
            console.error('启动摄像头失败:', error);
            this.showAlert('启动摄像头失败: ' + error.message, 'error');
            this.updateSystemStatus('错误');
        } finally {
            this.showLoading(false);
        }
    }

    stopCamera() {
        try {
            // 保存当前会话到历史记录
            if (this.currentSession && this.currentSession.detections.length > 0) {
                this.detectionHistory.push(this.currentSession);
                this.saveHistoryToStorage();
                console.log('检测会话已保存到历史记录');
            }
            
            // 停止所有媒体轨道
            if (this.cameraStream) {
                this.cameraStream.getTracks().forEach(track => {
                    track.stop();
                    console.log('停止媒体轨道:', track.kind);
                });
                this.cameraStream = null;
            }
            
            // 停止检测循环
            this.isDetecting = false;
            this.clearDetectionLoop();
            
            // 清理视频元素
            if (this.video) {
                this.video.srcObject = null;
                this.video.pause();
            }
            
            // 清理canvas
            if (this.videoCtx && this.videoCanvas) {
                this.videoCtx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
            }
            
            // 清理当前会话
            this.currentSession = null;
            
            // 更新UI状态
            this.updateCameraButtons();
            this.updateDetectionStatus('未启动');
            this.updateDroneCount(0);
            this.updateConfidence('-');
            
            this.showAlert('摄像头已停止', 'info');
            
        } catch (error) {
            console.error('停止摄像头时出错:', error);
            this.showAlert('停止摄像头时出错: ' + error.message, 'error');
        }
    }

    startDetectionLoop() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
        }
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
        }
        
        console.log('启动检测循环，间隔:', this.config.detectionInterval, 'ms');
        
        // 启动倒计时显示
        this.startCountdown();
        
        this.detectionInterval = setInterval(() => {
            if (this.isDetecting && this.video && this.video.readyState === 4) {
                console.log('执行检测循环');
                this.captureAndDetect();
                // 重新启动倒计时
                this.startCountdown();
            }
        }, this.config.detectionInterval);
    }

    clearDetectionLoop() {
        if (this.detectionInterval) {
            console.log('清除检测循环');
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
            this.countdownInterval = null;
        }
        // 清除倒计时显示
        this.updateCountdown('-');
    }

    async captureAndDetect() {
        if (!this.video || !this.videoCanvas || !this.videoCtx) {
            console.log('视频或canvas未准备就绪');
            return;
        }
        
        try {
            console.log('开始捕获视频帧进行检测');
            
            // 确保canvas尺寸匹配视频
            if (this.videoCanvas.width !== this.video.videoWidth || 
                this.videoCanvas.height !== this.video.videoHeight) {
                this.videoCanvas.width = this.video.videoWidth;
                this.videoCanvas.height = this.video.videoHeight;
            }
            
            // 将当前视频帧绘制到canvas
            this.videoCtx.drawImage(this.video, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
            
            // 将canvas转换为blob并发送到后端检测
            this.videoCanvas.toBlob(async (blob) => {
                if (!blob) {
                    console.error('无法创建图像blob');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', blob, 'video_frame.jpg');
                
                try {
                    console.log('发送检测请求到后端');
                    const response = await fetch('/api/detect-image', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const result = await response.json();
                    console.log('检测结果:', result);
                    
                    if (result.success) {
                        await this.handleVideoDetectionResult(result);
                    } else {
                        console.log('检测未发现目标');
                    }
                    
                } catch (error) {
                    console.error('检测请求失败:', error);
                    this.showAlert('检测请求失败: ' + error.message, 'error');
                }
            }, 'image/jpeg', 0.9);
            
        } catch (error) {
            console.error('捕获和检测失败:', error);
            this.showAlert('视频检测失败: ' + error.message, 'error');
        }
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // 激活音频上下文以确保警报声音可以播放
        await this.activateAudioContext();
        
        // 显示图片检测区域
        const imageDisplay = document.getElementById('imageDisplay');
        const imageCanvas = document.getElementById('imageCanvas');
        const ctx = imageCanvas.getContext('2d');
        
        // 创建图片对象并显示
        const img = new Image();
        img.onload = () => {
            imageCanvas.width = img.width;
            imageCanvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            if (imageDisplay) {
                imageDisplay.innerHTML = '';
                imageDisplay.appendChild(img);
            }
        };
        img.src = URL.createObjectURL(file);
        
        const formData = new FormData();
        formData.append('image', file);
        
        let loadingTimeout, timeoutId;
        
        try {
            this.showLoading(true);
            
            // 设置1秒后自动隐藏加载界面，提升用户体验
            loadingTimeout = setTimeout(() => {
                this.showLoading(false);
            }, 1000);
            
            // 设置较短的超时时间以提升用户体验
            const controller = new AbortController();
            timeoutId = setTimeout(() => controller.abort(), 10000); // 10秒超时
            
            const response = await fetch('/api/detect-image', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            clearTimeout(loadingTimeout);
            
            const result = await response.json();
            
            // 确保加载状态被隐藏
            this.showLoading(false);
            
            if (result.success) {
                // 在图片上绘制检测结果
                this.drawDetectionResults(ctx, result.detections || []);
                
                // 处理检测结果（包含统计更新和警报触发）
                await this.handleDetectionResult(result);
                
                // 根据检测结果显示不同的消息
                if (result.droneCount > 0) {
                    this.showAlert(`检测完成！发现 ${result.droneCount} 个无人机`, 'warning');
                } else {
                    this.showAlert('图片检测完成，未发现无人机', 'success');
                }
            } else {
                this.showAlert('检测失败: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('图片上传失败:', error);
            
            // 清除所有定时器并隐藏加载状态
            if (loadingTimeout) clearTimeout(loadingTimeout);
            if (timeoutId) clearTimeout(timeoutId);
            this.showLoading(false);
            
            if (error.name === 'AbortError') {
                this.showAlert('图片检测超时，请重试', 'error');
            } else {
                this.showAlert('图片上传失败: ' + error.message, 'error');
            }
        }
    }

    drawDetectionResults(ctx, detections) {
        if (!detections || detections.length === 0) {
            console.log('没有检测结果需要绘制');
            return;
        }
        
        console.log('绘制检测结果:', detections.length, '个目标');
        
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 3;
        ctx.fillStyle = '#ff0000';
        ctx.font = '16px Arial';
        
        detections.forEach((detection, index) => {
            const { bbox, confidence, class: className } = detection;
            
            if (!bbox || bbox.length !== 4) {
                console.error('无效的边界框数据:', bbox);
                return;
            }
            
            const [x, y, width, height] = bbox;
            
            // 绘制边界框
            ctx.strokeRect(x, y, width, height);
            
            // 绘制半透明背景
            ctx.fillStyle = 'rgba(255, 68, 68, 0.2)';
            ctx.fillRect(x, y, width, height);
            
            // 绘制标签背景
            const label = `${className || 'drone'} ${(confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = '#ff0000';
            ctx.fillRect(x, y - 25, textWidth + 10, 25);
            
            // 绘制标签文字
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + 5, y - 5);
            ctx.fillStyle = '#ff0000';
            
            console.log(`绘制目标 ${index + 1}: ${label} at (${x}, ${y}, ${width}, ${height})`);
        });
    }
    
    drawVideoDetectionResults(detections) {
        if (!this.videoCtx || !this.videoCanvas || !detections || detections.length === 0) {
            return;
        }
        
        console.log('在视频上绘制检测结果:', detections.length, '个目标');
        
        // 先清除之前的绘制
        this.videoCtx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
        
        // 重新绘制当前视频帧
        if (this.video) {
            this.videoCtx.drawImage(this.video, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
        }
        
        // 设置绘制样式
        this.videoCtx.strokeStyle = '#ff0000';
        this.videoCtx.lineWidth = 3;
        this.videoCtx.fillStyle = '#ff0000';
        this.videoCtx.font = '16px Arial';
        
        detections.forEach((detection, index) => {
            const { bbox, confidence, class: className } = detection;
            
            if (!bbox || bbox.length !== 4) {
                console.error('无效的边界框数据:', bbox);
                return;
            }
            
            const [x, y, width, height] = bbox;
            
            // 绘制边界框
            this.videoCtx.strokeRect(x, y, width, height);
            
            // 绘制半透明背景
            this.videoCtx.fillStyle = 'rgba(255, 68, 68, 0.2)';
            this.videoCtx.fillRect(x, y, width, height);
            
            // 绘制标签背景
            const label = `${className || 'drone'} ${(confidence * 100).toFixed(1)}%`;
            const textWidth = this.videoCtx.measureText(label).width;
            this.videoCtx.fillStyle = '#ff0000';
            this.videoCtx.fillRect(x, y - 25, textWidth + 10, 25);
            
            // 绘制标签文字
            this.videoCtx.fillStyle = '#ffffff';
            this.videoCtx.fillText(label, x + 5, y - 5);
            this.videoCtx.fillStyle = '#ff0000';
            
            console.log(`在视频上绘制目标 ${index + 1}: ${label} at (${x}, ${y}, ${width}, ${height})`);
        });
    }
    
    clearVideoDetectionResults() {
        if (!this.videoCtx || !this.videoCanvas) {
            return;
        }
        
        // 清除canvas并重新绘制视频帧
        this.videoCtx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
        
        if (this.video && this.video.readyState === 4) {
            this.videoCtx.drawImage(this.video, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
        }
    }

    async handleDetectionResult(result) {
        if (result.success && result.detections && result.detections.length > 0) {
            this.updateDetectionStats(result.detections);
            
            // 实时更新检测状态显示
            this.updateDetectionStatus('检测完成');
            this.updateDroneCount(result.detections.length);
            
            // 计算当前检测的平均置信度
            const avgConf = result.detections.reduce((sum, det) => sum + det.confidence, 0) / result.detections.length;
            this.updateConfidence(`${(avgConf * 100).toFixed(1)}%`);
            
            // 警报触发已移至updateUI函数中，基于检测次数变化
            
            this.updateUI();
        } else {
            this.stats.consecutiveDetections = 0;
            this.stats.droneCount = 0;
            
            // 实时更新检测状态显示
            this.updateDetectionStatus('检测完成');
            this.updateDroneCount(0);
            this.updateConfidence('-');
            
            this.updateUI();
        }
        
        // 显示检测结果
        this.displayDetectionResults(result.detections || []);
        
        // 更新检测信息显示
        this.updateDetectionInfo(result);
    }
    
    async handleVideoDetectionResult(result) {
        if (result.success && result.detections && result.detections.length > 0) {
            console.log('处理视频检测结果:', result.detections.length, '个目标');
            
            // 更新统计数据
            this.updateDetectionStats(result.detections);
            
            // 在视频上绘制检测结果
            this.drawVideoDetectionResults(result.detections);
            
            // 实时更新检测状态显示
            this.updateDetectionStatus('检测中');
            this.updateDroneCount(result.detections.length);
            
            // 计算当前检测的平均置信度
            const avgConf = result.detections.reduce((sum, det) => sum + det.confidence, 0) / result.detections.length;
            this.updateConfidence(`${(avgConf * 100).toFixed(1)}%`);
            
            // 警报触发已移至updateUI函数中，基于检测次数变化
            
            // 更新UI显示
            this.updateUI();
            
        } else {
            console.log('视频检测未发现目标');
            // 清除之前的检测框
            this.clearVideoDetectionResults();
            this.stats.consecutiveDetections = 0;
            this.stats.droneCount = 0;
            
            // 实时更新检测状态显示
            this.updateDetectionStatus('检测中');
            this.updateDroneCount(0);
            this.updateConfidence('-');
            
            this.updateUI();
        }
    }
    
    updateDetectionStats(detections) {
        this.stats.totalDetections++;
        this.stats.consecutiveDetections++;
        this.stats.lastDetectionTime = new Date();
        this.stats.droneCount = detections.length;
        
        // 累计总无人机检测数
        this.stats.totalDroneCount += detections.length;
        
        // 计算平均置信度 - 基于检测到的无人机数量
        const totalConfidence = detections.reduce((sum, det) => sum + det.confidence, 0);
        this.stats.confidenceSum += totalConfidence;
        // 只有当检测到无人机时才计算平均置信度
        this.stats.avgConfidence = this.stats.totalDroneCount > 0 ? 
            this.stats.confidenceSum / this.stats.totalDroneCount : 0;
        
        // 添加到检测历史
        const historyItem = {
            timestamp: new Date(),
            droneCount: detections.length,
            avgConfidence: totalConfidence / detections.length,
            detections: detections.map(det => ({
                class: det.class || 'drone',
                confidence: det.confidence,
                bbox: det.bbox,
                timestamp: new Date()
            }))
        };
        
        this.detectionHistory.unshift(historyItem);
        
        // 限制历史记录数量
        if (this.detectionHistory.length > 100) {
            this.detectionHistory = this.detectionHistory.slice(0, 100);
        }
        
        // 保存检测历史到本地存储
        this.saveHistoryToStorage();
        
        console.log('更新统计数据:', {
            total: this.stats.totalDetections,
            totalDrones: this.stats.totalDroneCount,
            current: this.stats.droneCount,
            avgConf: this.stats.avgConfidence.toFixed(2)
        });
    }

    displayDetectionResults(detections) {
        const resultsContainer = document.getElementById('detectionResults');
        if (!resultsContainer) return;
        
        if (detections.length === 0) {
            resultsContainer.innerHTML = '<p class="no-detection">未检测到无人机</p>';
            return;
        }
        
        let html = `<h3>检测到 ${detections.length} 个无人机:</h3>`;
        detections.forEach((detection, index) => {
            html += `
                <div class="detection-item">
                    <span class="detection-label">无人机 ${index + 1}</span>
                    <span class="confidence">置信度: ${(detection.confidence * 100).toFixed(1)}%</span>
                    <span class="position">位置: (${detection.x}, ${detection.y})</span>
                </div>
            `;
        });
        
        resultsContainer.innerHTML = html;
    }

    updateDetectionInfo(result) {
        const infoContainer = document.getElementById('detectionInfo');
        if (!infoContainer) return;
        
        const timestamp = new Date(result.timestamp).toLocaleTimeString();
        
        let html = `
            <div class="info-item">
                <span class="info-label">检测时间:</span>
                <span class="info-value">${timestamp}</span>
            </div>
            <div class="info-item">
                <span class="info-label">无人机数量:</span>
                <span class="info-value">${result.droneCount || 0}</span>
            </div>
            <div class="info-item">
                <span class="info-label">连续检测:</span>
                <span class="info-value">${result.stats?.consecutiveDetections || 0} 次</span>
            </div>
        `;
        
        if (result.geminiAnalysis) {
            html += `
                <div class="info-item gemini-analysis">
                    <span class="info-label">AI分析:</span>
                    <span class="info-value">${result.geminiAnalysis.analysis}</span>
                </div>
            `;
        }
        
        infoContainer.innerHTML = html;
    }

    async triggerShortAlert() {
        console.log('触发短警报 - 音频警报:', this.config.soundAlerts);
        
        this.showAlert('检测到无人机活动！', 'warning');
        // 播放噪声警报音
        if (this.config.soundAlerts) {
            await this.playNoiseAlarm();
        }
        
        if (this.config.desktopNotifications && 'Notification' in window) {
            if (Notification.permission === 'granted') {
                new Notification('无人机检测警报', {
                    body: '检测到无人机活动',
                    icon: '/favicon.ico'
                });
            } else if (Notification.permission === 'default') {
                // 请求通知权限
                const permission = await Notification.requestPermission();
                if (permission === 'granted') {
                    new Notification('无人机检测警报', {
                        body: '检测到无人机活动',
                        icon: '/favicon.ico'
                    });
                }
            }
        }
    }

    async triggerLongAlert(geminiAnalysis) {
        console.log('触发长警报 - 音频警报:', this.config.soundAlerts);
        
        const message = geminiAnalysis ? 
            `多无人机警告！AI确认: ${geminiAnalysis.analysis}` : 
            '检测到多个无人机，触发高级警报！';
        
        this.showAlert(message, 'danger');
        
        // 播放噪声警报音
        if (this.config.soundAlerts) {
            await this.playNoiseAlarm();
        }
        
        // 桌面通知
        if (this.config.desktopNotifications && 'Notification' in window) {
            if (Notification.permission === 'granted') {
                new Notification('无人机检测警报', {
                    body: message,
                    icon: '/static/drone-icon.png'
                });
            } else if (Notification.permission !== 'denied') {
                const permission = await Notification.requestPermission();
                if (permission === 'granted') {
                    new Notification('无人机检测警报', {
                        body: message,
                        icon: '/static/drone-icon.png'
                    });
                }
            }
        }
    }

    updateSystemStatus(status) {
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            const statusText = statusElement.querySelector('.status-text');
            const statusDot = statusElement.querySelector('.status-dot');
            
            if (statusText) statusText.textContent = `系统${status}`;
            if (statusDot) statusDot.className = `status-dot ${status === '在线' ? 'online' : 'offline'}`;
        }
    }

    updateUI() {
        // 更新统计数据显示
        const totalDetectionsEl = document.getElementById('totalDetections');
        const droneDetectionsEl = document.getElementById('droneDetections');
        const avgConfidenceEl = document.getElementById('avgConfidence');
        const lastDetectionEl = document.getElementById('lastDetectionTime');
        
        if (totalDetectionsEl) {
            const currentCount = this.stats.totalDetections > 0 ? this.stats.totalDetections : 0;
            totalDetectionsEl.textContent = currentCount > 0 ? currentCount : '--';
        }
        
        // 音频播放逻辑已移至updateDroneCount函数中处理
        if (droneDetectionsEl) {
            // 使用累计的总无人机检测数
            droneDetectionsEl.textContent = this.stats.totalDroneCount > 0 ? this.stats.totalDroneCount : '--';
        }
        if (avgConfidenceEl) {
            const avgConf = this.stats.avgConfidence > 0 ? (this.stats.avgConfidence * 100).toFixed(1) + '%' : '--';
            avgConfidenceEl.textContent = avgConf;
        }
        if (lastDetectionEl) {
            const lastTime = this.stats.lastDetectionTime ? 
                new Date(this.stats.lastDetectionTime).toLocaleString('zh-CN', {
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                }) : '--';
            lastDetectionEl.textContent = lastTime;
        }
        
        // 更新实时检测状态
        this.updateDetectionStatus(this.isDetecting ? '检测中' : '未启动');
        
        // 更新当前检测的无人机数量
        this.updateDroneCount(this.stats.droneCount || 0);
        
        // 更新当前检测的置信度
        const currentConfidence = this.stats.droneCount > 0 && this.stats.avgConfidence ? 
            `${(this.stats.avgConfidence * 100).toFixed(1)}%` : '-';
        this.updateConfidence(currentConfidence);
        
        // 更新摄像头按钮状态
        this.updateCameraButtons();
        
        // 更新检测历史数量显示
        const historyCountEl = document.getElementById('historyCount');
        if (historyCountEl) {
            historyCountEl.textContent = this.detectionHistory.length;
        }
        
        console.log('UI已更新:', {
            total: this.stats.totalDetections,
            totalDrones: this.stats.totalDroneCount,
            current: this.stats.droneCount,
            avgConf: (this.stats.avgConfidence || 0).toFixed(2),
            historyCount: this.detectionHistory.length
        });
    }
    
    updateDetectionStatus(status) {
        const statusEl = document.getElementById('detectionStatus');
        if (statusEl) {
            statusEl.textContent = status;
            statusEl.className = `status ${status === '检测中' ? 'active' : 'inactive'}`;
        }
    }
    
    updateDroneCount(count) {
        const countEl = document.getElementById('droneCount');
        if (countEl) {
            countEl.textContent = count;
            countEl.className = count > 0 ? 'count alert' : 'count';
        }
        
        // 音频播放逻辑：当无人机数量变化时立即播放音频
        console.log('🔍 updateDroneCount 调用:', {
            count: count,
            previousCount: this.previousDroneCount,
            soundAlerts: this.config.soundAlerts,
            countChanged: count !== this.previousDroneCount,
            isAlarmPlaying: this.isAlarmPlaying,
            audioContextState: this.audioContext ? this.audioContext.state : 'null'
        });
        
        if (this.config.soundAlerts && count !== this.previousDroneCount) {
            if (count > 0) {
                console.log('🚨 无人机数量变化:', this.previousDroneCount, '->', count, '立即播放警报');
                console.log('🎵 开始播放音频流程...');
                this.playNoiseAlarm().catch(error => {
                    console.error('❌ 播放警报音失败:', error);
                    this.showAlert('音频播放失败: ' + error.message, 'error');
                });
            } else if (this.previousDroneCount > 0 && count === 0) {
                console.log('✅ 无人机数量归零，停止警报');
                this.stopAlarm();
            }
        } else {
            console.log('⚠️ 音频播放条件不满足:', {
                soundAlerts: this.config.soundAlerts,
                countChanged: count !== this.previousDroneCount,
                reason: !this.config.soundAlerts ? '音频警报已禁用' : '数量未变化'
            });
        }
        
        // 更新之前的数量
        this.previousDroneCount = count;
    }
    
    updateConfidence(confidence) {
        const confidenceEl = document.getElementById('confidence');
        if (confidenceEl) {
            confidenceEl.textContent = confidence;
        }
    }
    
    updateCountdown(countdown) {
        const countdownEl = document.getElementById('countdown');
        if (countdownEl) {
            countdownEl.textContent = countdown;
        }
    }
    
    startCountdown() {
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
        }
        
        let remainingTime = Math.floor(this.config.detectionInterval / 1000); // 转换为秒
        this.updateCountdown(remainingTime + 's');
        
        this.countdownInterval = setInterval(() => {
            remainingTime--;
            if (remainingTime > 0) {
                this.updateCountdown(remainingTime + 's');
            } else {
                this.updateCountdown('检测中...');
                clearInterval(this.countdownInterval);
                this.countdownInterval = null;
            }
        }, 1000);
    }

    updateCameraButtons() {
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        
        if (startBtn && stopBtn) {
            if (this.isDetecting) {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-video"></i> 检测中...';
                startBtn.classList.add('opacity-50', 'cursor-not-allowed');
                stopBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            } else {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                startBtn.innerHTML = '<i class="fas fa-video"></i> 启动摄像头';
                startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }
        }
    }

    showConfigModal() {
        const modal = document.getElementById('configModal');
        
        // 填充当前配置
        document.getElementById('confidenceThreshold').value = this.config.confidenceThreshold;
        document.getElementById('detectionInterval').value = this.config.detectionInterval;
        document.getElementById('soundAlerts').checked = this.config.soundAlerts !== false;
        document.getElementById('desktopNotifications').checked = this.config.desktopNotifications !== false;
        
        console.log('显示配置模态框，当前配置:', this.config);
        modal.classList.remove('hidden');
    }

    async showHistoryModal() {
        const modal = document.getElementById('historyModal');
        
        // 填充检测历史数据
        const historyList = document.getElementById('historyList');
        if (historyList) {
            if (this.detectionHistory.length === 0) {
                historyList.innerHTML = '<div class="no-history"><p>暂无检测历史</p></div>';
            } else {
                const historyHTML = this.detectionHistory.map((item, index) => {
                    const timeStr = item.timestamp.toLocaleString('zh-CN');
                    const confStr = (item.avgConfidence * 100).toFixed(1);
                    return `
                        <div class="history-item" data-index="${index}">
                            <div class="history-header">
                                <span class="history-time">${timeStr}</span>
                                <span class="history-count">${item.droneCount} 个目标</span>
                            </div>
                            <div class="history-details">
                                <span class="history-confidence">平均置信度: ${confStr}%</span>
                                <span class="history-detections">${item.detections.length} 次检测</span>
                            </div>
                        </div>
                    `;
                }).join('');
                
                historyList.innerHTML = `
                    <div class="history-summary">
                        <p>总计 ${this.detectionHistory.length} 条记录</p>
                    </div>
                    <div class="history-items">
                        ${historyHTML}
                    </div>
                `;
            }
        }
        
        console.log('显示历史记录模态框，记录数量:', this.detectionHistory.length);
        modal.classList.remove('hidden');
    }

    closeModal(modal) {
        modal.classList.add('hidden');
    }

    async saveConfig() {
        try {
            const confidenceInput = document.getElementById('confidenceThreshold');
            const intervalInput = document.getElementById('detectionInterval');
            const soundAlertsInput = document.getElementById('soundAlerts');
            const desktopNotificationsInput = document.getElementById('desktopNotifications');
            
            // 更新配置
            const newConfig = { ...this.config };
            
            if (confidenceInput) {
                newConfig.confidenceThreshold = parseFloat(confidenceInput.value);
                if (newConfig.confidenceThreshold < 0.1 || newConfig.confidenceThreshold > 1.0) {
                    throw new Error('置信度阈值必须在0.1-1.0之间');
                }
            }
            
            if (intervalInput) {
                const intervalMs = parseInt(intervalInput.value);
                if (intervalMs < 1000 || intervalMs > 60000) {
                    throw new Error('检测间隔必须在1000-60000毫秒之间');
                }
                newConfig.detectionInterval = intervalMs;
            }
            
            if (soundAlertsInput) {
                newConfig.soundAlerts = soundAlertsInput.checked;
            }
            
            if (desktopNotificationsInput) {
                newConfig.desktopNotifications = desktopNotificationsInput.checked;
            }
            
            // 保存到本地存储
            localStorage.setItem('droneDetectionConfig', JSON.stringify(newConfig));
            
            // 应用新配置
            const oldInterval = this.config.detectionInterval;
            this.config = newConfig;
            
            // 如果检测间隔改变且正在检测，重启检测循环
            if (this.isDetecting && oldInterval !== newConfig.detectionInterval) {
                this.clearDetectionLoop();
                this.startDetectionLoop();
            }
            
            console.log('配置已保存:', this.config);
            this.showAlert('配置保存成功', 'success');
            this.closeModal(document.getElementById('configModal'));
            
        } catch (error) {
            console.error('保存配置失败:', error);
            this.showAlert('保存配置失败: ' + error.message, 'error');
        }
    }

    async clearHistory() {
        try {
            if (this.detectionHistory.length === 0) {
                this.showAlert('没有历史记录需要清除', 'info');
                return;
            }
            
            // 确认对话框
            if (!confirm(`确定要清除所有 ${this.detectionHistory.length} 条检测历史记录吗？此操作不可撤销。`)) {
                return;
            }
            
            // 清除本地历史记录
            this.detectionHistory = [];
            
            // 清除本地存储
            localStorage.removeItem('droneDetectionHistory');
            
            // 重置本地统计
            this.stats.totalDetections = 0;
            this.stats.consecutiveDetections = 0;
            this.stats.confidenceSum = 0;
            this.stats.avgConfidence = 0;
            
            // 更新UI
            this.updateUI();
            
            console.log('历史记录已清除');
            this.showAlert('历史记录已清除', 'success');
            
            // 刷新历史模态框显示
            const modal = document.getElementById('historyModal');
            if (modal && !modal.classList.contains('hidden')) {
                this.showHistoryModal();
            }
            
        } catch (error) {
            console.error('清除历史记录失败:', error);
            this.showAlert('清除历史记录失败: ' + error.message, 'error');
        }
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) {
            console.log(`Alert (${type}): ${message}`);
            return;
        }
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <span class="alert-message">${message}</span>
            <button class="alert-close" onclick="this.parentElement.remove()">&times;</button>
        `;
        
        alertContainer.appendChild(alert);
        
        // 自动移除警报
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }

    showLoading(show = true) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            if (show) {
                loadingOverlay.classList.remove('hidden');
            } else {
                loadingOverlay.classList.add('hidden');
            }
        }
    }
    
    // 加载保存的配置和历史记录
    loadSavedData() {
        try {
            // 加载配置
            const savedConfig = localStorage.getItem('droneDetectionConfig');
            if (savedConfig) {
                const config = JSON.parse(savedConfig);
                this.config = { ...this.config, ...config };
                console.log('已加载保存的配置:', this.config);
            }
            
            // 加载历史记录
            const savedHistory = localStorage.getItem('droneDetectionHistory');
            if (savedHistory) {
                const history = JSON.parse(savedHistory);
                // 恢复Date对象
                this.detectionHistory = history.map(item => ({
                    ...item,
                    timestamp: new Date(item.timestamp),
                    detections: item.detections.map(det => ({
                        ...det,
                        timestamp: new Date(det.timestamp)
                    }))
                }));
                
                // 重新计算统计数据
                this.recalculateStats();
                console.log('已加载历史记录:', this.detectionHistory.length, '条');
            }
            
        } catch (error) {
            console.error('加载保存的数据失败:', error);
        }
    }
    
    // 重新计算统计数据
    recalculateStats() {
        this.stats.totalDetections = 0;
        this.stats.confidenceSum = 0;
        let totalDroneCount = 0;
        
        this.detectionHistory.forEach(session => {
            this.stats.totalDetections += session.detections.length;
            totalDroneCount += session.droneCount;
            session.detections.forEach(det => {
                this.stats.confidenceSum += det.confidence;
            });
        });
        
        this.stats.avgConfidence = this.stats.totalDetections > 0 ? 
            this.stats.confidenceSum / this.stats.totalDetections : 0;
        
        // 更新最后检测时间
        if (this.detectionHistory.length > 0) {
            const lastSession = this.detectionHistory[this.detectionHistory.length - 1];
            if (lastSession.detections.length > 0) {
                this.stats.lastDetectionTime = lastSession.detections[lastSession.detections.length - 1].timestamp;
            }
        }
        
        console.log('统计数据已重新计算:', this.stats);
    }
    
    // 保存历史记录到本地存储
    saveHistoryToStorage() {
        try {
            localStorage.setItem('droneDetectionHistory', JSON.stringify(this.detectionHistory));
        } catch (error) {
            console.error('保存检测历史失败:', error);
        }
    }
    
    // 导出历史记录
    exportHistory() {
        try {
            if (this.detectionHistory.length === 0) {
                this.showAlert('没有历史记录可以导出', 'info');
                return;
            }
            
            // 准备导出数据
            const exportData = {
                exportTime: new Date().toISOString(),
                totalSessions: this.detectionHistory.length,
                totalDetections: this.stats.totalDetections,
                avgConfidence: this.stats.avgConfidence,
                history: this.detectionHistory.map(session => ({
                    timestamp: session.timestamp.toISOString(),
                    droneCount: session.droneCount,
                    avgConfidence: session.avgConfidence,
                    detections: session.detections.map(det => ({
                        timestamp: det.timestamp.toISOString(),
                        confidence: det.confidence,
                        bbox: det.bbox
                    }))
                }))
            };
            
            // 创建下载链接
            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `drone_detection_history_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            URL.revokeObjectURL(url);
            
            console.log('历史记录已导出');
            this.showAlert('历史记录导出成功', 'success');
            
        } catch (error) {
            console.error('导出历史记录失败:', error);
            this.showAlert('导出历史记录失败: ' + error.message, 'error');
        }
    }
    
    // 测试音频功能
    testAudio() {
        console.log('测试音频功能被调用');
        
        // 检查音频权限
        if (!this.checkAudioPermissions()) {
            this.showAlert('音频权限检查失败，请确保浏览器允许音频播放', 'error');
            return;
        }
        
        // 激活音频上下文（需要用户交互）
        this.activateAudioContext();
        
        // 播放测试音频
        this.playNoiseAlarm();
        
        // 显示测试提示
        this.showAlert('正在播放测试警报音频...', 'info');
        
        // 3秒后自动停止
        setTimeout(() => {
            this.stopAlarm();
            this.showAlert('测试音频播放完成', 'success');
        }, 3000);
    }
}

// 初始化系统
document.addEventListener('DOMContentLoaded', () => {
    window.droneSystem = new DroneDetectionSystem();
    
    // 请求桌面通知权限
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

// 页面卸载时清理资源
window.addEventListener('beforeunload', () => {
    if (window.droneSystem) {
        window.droneSystem.stopCamera();
    }
});