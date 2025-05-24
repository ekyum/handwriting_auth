//
//  ContentView.swift
//  Signature
//
//  Created by haro on 5/4/25.
//
import SwiftUI
import PencilKit
import UniformTypeIdentifiers

// MARK: - Data Models
struct SignatureData: Codable {
    let writerCode: String
    var samples: [SignatureSample]
}

struct SignatureSample: Codable {
    var strokes: [StrokeData]
    let timestamp: Date
}

struct StrokeData: Codable {
    var points: [StrokePointData]
    let color: String
}

struct StrokePointData: Codable {
    let x: CGFloat
    let y: CGFloat
    let force: CGFloat
    let timeOffset: TimeInterval
    let size: [CGFloat]
    let opacity: CGFloat
    let azimuth: CGFloat
    let altitude: CGFloat
}

// MARK: - View Model
class SignatureViewModel: ObservableObject {
    @Published var canvasView = PKCanvasView()
    @Published var isRecording = false
    @Published var signatureData = SignatureData(writerCode: UUID().uuidString, samples: [])
    @Published var isExporting = false
    
    let toolPicker = PKToolPicker()
    
    private var currentSample: SignatureSample?
    
    func startRecording() {
        currentSample = SignatureSample(strokes: [], timestamp: Date())
        isRecording = true
    }
    
    func endRecording() {
        isRecording = false
        processCurrentDrawing()
    }
    
    func reset() {
        canvasView.drawing = PKDrawing()
        signatureData.samples.removeAll()
    }
    
    private func processCurrentDrawing() {
        guard var sample = currentSample else { return }
        
        for stroke in canvasView.drawing.strokes {
            var strokeData = StrokeData(
                points: [],
                color: stroke.ink.color.hexString
            )
            
            for point in stroke.path {
                let pointData = StrokePointData(
                    x: point.location.x,
                    y: point.location.y,
                    force: point.force,
                    timeOffset: point.timeOffset,
                    size: [point.size.width, point.size.height],
                    opacity: point.opacity,
                    azimuth: point.azimuth,
                    altitude: point.altitude
                )
                strokeData.points.append(pointData)
            }
            
            sample.strokes.append(strokeData)
        }
        
        signatureData.samples.append(sample)
        canvasView.drawing = PKDrawing()
    }
    
    func exportData() -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        encoder.dateEncodingStrategy = .iso8601
        
        do {
            return try encoder.encode(signatureData)
        } catch {
            print("Encoding error: \(error)")
            return nil
        }
    }
}

// MARK: - Canvas View Wrapper
struct CanvasView: UIViewRepresentable {
    @Binding var canvasView: PKCanvasView
    @Binding var isRecording: Bool
    let toolPicker: PKToolPicker
    
    func makeUIView(context: Context) -> PKCanvasView {
        canvasView.drawingPolicy = .anyInput
        canvasView.backgroundColor = .clear
        canvasView.isOpaque = false
        
        toolPicker.setVisible(true, forFirstResponder: canvasView)
        toolPicker.addObserver(canvasView)
        canvasView.becomeFirstResponder()
        
        return canvasView
    }
    
    func updateUIView(_ uiView: PKCanvasView, context: Context) {
        uiView.drawingPolicy = isRecording ? .anyInput : .pencilOnly
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, PKCanvasViewDelegate {
        var parent: CanvasView
        
        init(_ parent: CanvasView) {
            self.parent = parent
        }
    }
}

// MARK: - File Document
struct JSONDocument: FileDocument {
    static var readableContentTypes: [UTType] = [.json]
    var data: Data
    
    init(data: Data) {
        self.data = data
    }
    
    init(configuration: ReadConfiguration) throws {
        data = Data()
    }
    
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

// MARK: - Main View
struct SignatureCaptureView: View {
    @StateObject private var viewModel = SignatureViewModel()
    
    var body: some View {
        VStack {
            // Sample Counter
            Text("Samples Collected: \(viewModel.signatureData.samples.count)")
                .font(.headline)
                .padding()
            
            // Drawing Canvas
            CanvasView(
                canvasView: $viewModel.canvasView,
                isRecording: $viewModel.isRecording,
                toolPicker: viewModel.toolPicker
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(12)
            .padding()
            
            // Control Buttons
            HStack(spacing: 20) {
                ControlButton(
                    label: viewModel.isRecording ? "End" : "Start",
                    color: viewModel.isRecording ? .red : .green
                ) {
                    viewModel.isRecording ? viewModel.endRecording() : viewModel.startRecording()
                }
                
                ControlButton(label: "Reset", color: .blue) {
                    viewModel.reset()
                }
                
                ControlButton(label: "Export", color: .purple) {
                    viewModel.isExporting = true
                }
            }
            .padding(.bottom)
        }
        .fileExporter(
            isPresented: $viewModel.isExporting,
            document: JSONDocument(data: viewModel.exportData() ?? Data()),
            contentType: .json,
            defaultFilename: "signature_data"
        ) { result in
            handleExportResult(result)
        }
    }
    
    private func handleExportResult(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            print("Exported to: \(url.path)")
        case .failure(let error):
            print("Export error: \(error.localizedDescription)")
        }
    }
}

struct ControlButton: View {
    let label: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 18, weight: .semibold))
                .foregroundColor(.white)
                .padding()
                .frame(minWidth: 100)
                .background(color)
                .cornerRadius(8)
                .shadow(radius: 2)
        }
    }
}

// MARK: - Color Extension
extension UIColor {
    var hexString: String {
        var r: CGFloat = 0
        var g: CGFloat = 0
        var b: CGFloat = 0
        var a: CGFloat = 0
        
        getRed(&r, green: &g, blue: &b, alpha: &a)
        return String(
            format: "#%02lX%02lX%02lX%02lX",
            lroundf(Float(r * 255)),
            lroundf(Float(g * 255)),
            lroundf(Float(b * 255)),
            lroundf(Float(a * 255))
        )
    }
}

// MARK: - Preview
struct SignatureCaptureView_Previews: PreviewProvider {
    static var previews: some View {
        SignatureCaptureView()
    }
}

#Preview {
    SignatureCaptureView()
        .modelContainer(for: Item.self, inMemory: true)
}
