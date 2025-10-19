const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// 配置
const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY = 'AIzaSyD-YpZjl8o0f5vfpYd3eIYxHlRGtTnV9Ms';
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));
app.use(express.static(path.join(__dirname, '..')));

// 配置multer用于文件上传
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// 系统配置
let systemConfig = {
    detectionInterval: 2000,
    alertThreshold: 3,
    multiDroneThreshold: 2,
    confidenceThreshold: 0.5
};

// 检测统计
let detectionStats = {
    totalDetections: 0,
    consecutiveDetections: 0,
    lastDetectionTime: null,
    droneCount: 0,
    detectionHistory: []
};

// YOLO26检测类
class YOLO26Detector {
    constructor() {
        this.pythonPath = 'python';
        this.scriptPath = path.join(__dirname, 'yolo_detector.py');
    }

    async detectImage(imagePath) {
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn(this.pythonPath, [
                this.scriptPath,
                'detect',
                imagePath,
                systemConfig.confidenceThreshold.toString()
            ]);

            let output = '';
            let errorOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        // 提取最后一行作为JSON结果，忽略其他输出
                        const lines = output.trim().split('\n');
                        const jsonLine = lines[lines.length - 1];
                        const result = JSON.parse(jsonLine);
                        resolve(result);
                    } catch (error) {
                        console.error('解析检测结果失败:', error);
                        console.error('Python输出:', output);
                        reject(new Error('Failed to parse detection result'));
                    }
                } else {
                    reject(new Error(`Detection failed: ${errorOutput}`));
                }
            });
        });
    }
}

const detector = new YOLO26Detector();

// Gemini AI分析
async function analyzeWithGemini(imagePath, droneCount) {
    try {
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
        
        const imageData = fs.readFileSync(imagePath);
        const base64Image = imageData.toString('base64');
        
        const prompt = `请分析这张图片中是否真的存在${droneCount}个或更多无人机。请仔细观察图片中的物体，确认它们是否为无人机。请用中文回答，格式：{"confirmed": true/false, "actualCount": 数量, "analysis": "分析说明"}`;
        
        const result = await model.generateContent([
            prompt,
            {
                inlineData: {
                    data: base64Image,
                    mimeType: "image/jpeg"
                }
            }
        ]);
        
        const response = await result.response;
        const text = response.text();
        
        try {
            return JSON.parse(text);
        } catch {
            return {
                confirmed: false,
                actualCount: 0,
                analysis: "AI分析失败"
            };
        }
    } catch (error) {
        console.error('Gemini分析错误:', error);
        return {
            confirmed: false,
            actualCount: 0,
            analysis: "AI分析服务不可用"
        };
    }
}

// 处理检测结果
function processDetectionResult(detections) {
    const currentTime = new Date();
    const droneCount = detections.length;
    
    // 更新统计信息
    detectionStats.totalDetections++;
    detectionStats.droneCount = droneCount;
    detectionStats.lastDetectionTime = currentTime;
    
    if (droneCount > 0) {
        detectionStats.consecutiveDetections++;
    } else {
        detectionStats.consecutiveDetections = 0;
    }
    
    // 添加到历史记录
    detectionStats.detectionHistory.unshift({
        timestamp: currentTime,
        droneCount: droneCount,
        detections: detections
    });
    
    // 保持历史记录在100条以内
    if (detectionStats.detectionHistory.length > 100) {
        detectionStats.detectionHistory = detectionStats.detectionHistory.slice(0, 100);
    }
    
    return {
        droneCount,
        shouldTriggerShortAlert: detectionStats.consecutiveDetections >= systemConfig.alertThreshold,
        shouldTriggerLongAlert: droneCount >= systemConfig.multiDroneThreshold,
        stats: detectionStats
    };
}

// API路由

// 获取系统状态
app.get('/api/status', (req, res) => {
    res.json({
        status: 'online',
        config: systemConfig,
        stats: detectionStats
    });
});

// 更新系统配置
app.post('/api/config', (req, res) => {
    const { detectionInterval, alertThreshold, multiDroneThreshold, confidenceThreshold } = req.body;
    
    if (detectionInterval) systemConfig.detectionInterval = detectionInterval;
    if (alertThreshold) systemConfig.alertThreshold = alertThreshold;
    if (multiDroneThreshold) systemConfig.multiDroneThreshold = multiDroneThreshold;
    if (confidenceThreshold) systemConfig.confidenceThreshold = confidenceThreshold;
    
    res.json({ success: true, config: systemConfig });
});

// 图片上传检测
app.post('/api/detect-image', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: '没有上传图片' });
        }
        
        // 保存临时图片文件
        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }
        
        const tempImagePath = path.join(tempDir, `temp_${Date.now()}.jpg`);
        fs.writeFileSync(tempImagePath, req.file.buffer);
        
        // 执行检测
        const pythonResult = await detector.detectImage(tempImagePath);
        
        // 处理Python脚本返回的结果
        let detections = [];
        if (pythonResult.success && pythonResult.detections) {
            detections = pythonResult.detections;
        }
        
        const result = processDetectionResult(detections);
        
        // 如果检测到多个无人机，使用Gemini进行二次确认
        let geminiAnalysis = null;
        if (result.shouldTriggerLongAlert) {
            geminiAnalysis = await analyzeWithGemini(tempImagePath, result.droneCount);
        }
        
        // 清理临时文件
        fs.unlinkSync(tempImagePath);
        
        // 广播检测结果
        io.emit('detection-result', {
            ...result,
            detections,
            geminiAnalysis,
            timestamp: new Date()
        });
        
        res.json({
            success: true,
            droneCount: result.droneCount,
            detections,
            stats: result.stats,
            shouldTriggerShortAlert: result.shouldTriggerShortAlert,
            shouldTriggerLongAlert: result.shouldTriggerLongAlert,
            geminiAnalysis,
            inferenceTime: pythonResult.inference_time_ms || 0,
            timestamp: new Date()
        });
        
    } catch (error) {
        console.error('图片检测错误:', error);
        res.status(500).json({ error: '检测失败', details: error.message });
    }
});

// 获取检测历史
app.get('/api/history', (req, res) => {
    res.json({
        history: detectionStats.detectionHistory,
        totalCount: detectionStats.detectionHistory.length
    });
});

// 清除检测历史
app.delete('/api/history', (req, res) => {
    detectionStats.detectionHistory = [];
    detectionStats.totalDetections = 0;
    detectionStats.consecutiveDetections = 0;
    res.json({ success: true });
});

// WebSocket连接处理
io.on('connection', (socket) => {
    console.log('客户端已连接:', socket.id);
    
    // 发送当前状态
    socket.emit('system-status', {
        config: systemConfig,
        stats: detectionStats
    });
    
    // 处理摄像头流检测请求
    socket.on('start-camera-detection', () => {
        console.log('开始摄像头检测');
        socket.emit('camera-detection-started');
    });
    
    socket.on('stop-camera-detection', () => {
        console.log('停止摄像头检测');
        socket.emit('camera-detection-stopped');
    });
    
    // 处理摄像头帧数据
    socket.on('camera-frame', async (frameData) => {
        try {
            // 将base64图片数据保存为临时文件
            const tempDir = path.join(__dirname, 'temp');
            if (!fs.existsSync(tempDir)) {
                fs.mkdirSync(tempDir, { recursive: true });
            }
            
            const tempImagePath = path.join(tempDir, `frame_${Date.now()}.jpg`);
            const base64Data = frameData.replace(/^data:image\/jpeg;base64,/, '');
            fs.writeFileSync(tempImagePath, base64Data, 'base64');
            
            // 执行检测
            const detections = await detector.detectImage(tempImagePath);
            const result = processDetectionResult(detections);
            
            // 如果检测到多个无人机，使用Gemini进行二次确认
            let geminiAnalysis = null;
            if (result.shouldTriggerLongAlert) {
                geminiAnalysis = await analyzeWithGemini(tempImagePath, result.droneCount);
            }
            
            // 清理临时文件
            fs.unlinkSync(tempImagePath);
            
            // 发送检测结果
            socket.emit('detection-result', {
                ...result,
                detections,
                geminiAnalysis,
                timestamp: new Date()
            });
            
        } catch (error) {
            console.error('摄像头帧检测错误:', error);
            socket.emit('detection-error', { error: error.message });
        }
    });
    
    socket.on('disconnect', () => {
        console.log('客户端已断开连接:', socket.id);
    });
});

// 启动服务器
server.listen(PORT, () => {
    console.log(`无人机检测服务器运行在 http://localhost:${PORT}`);
    console.log('WebSocket服务已启动');
});

// 优雅关闭
process.on('SIGINT', () => {
    console.log('\n正在关闭服务器...');
    server.close(() => {
        console.log('服务器已关闭');
        process.exit(0);
    });
});