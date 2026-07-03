@preconcurrency import CoreML
import CryptoKit
import FluidAudio
import Foundation

/// Store for the Omi Med STT v1 CoreML bundle — an English medical
/// fine-tune of NVIDIA Parakeet TDT 0.6B v2 (`omi-health/omi-med-stt-v1`).
///
/// The bundle is architecturally identical to FluidAudio's stock v2 build
/// (same tokenizer, blank id 1024, same component I/O contract), so the
/// loaded models drive the shared TDT `AsrManager` as `AsrModelVersion.v2`.
/// The compiled CoreML artifacts are published at
/// `huggingface.co/cmsha/omi-med-stt-v1-coreml` (converted offline from the
/// upstream `.nemo`; recipe in this folder's README) and download into
/// ``modelDirectory()`` on first use, mirroring the stock builds' behavior.
///
/// Download *and* load deliberately bypass FluidAudio's `DownloadUtils` /
/// `AsrModels.downloadAndLoad`: that path only knows FluidAudio's own repos,
/// and its corrupt-cache recovery re-downloads the *stock* v2 weights, which
/// would silently replace the medical fine-tune. A missing or partial
/// install throws instead of ever falling back to stock weights.
public enum OmiMedParakeetModel {

    /// Source repository on HuggingFace for the compiled bundle.
    public static let huggingFaceRepo = "cmsha/omi-med-stt-v1-coreml"

    /// Leaf directory under FluidAudio's models base holding the compiled
    /// bundle. Sibling of the stock `parakeet-tdt-0.6b-v2` cache so `models
    /// clear`-style maintenance sees every speech model in one place.
    public static let folderName = "omi-med-stt-v1-coreml"

    /// Compiled component bundles, mirroring FluidAudio's v2 layout
    /// (`ModelNames.ASR`): split preprocessor/encoder frontend, RNNT
    /// prediction network, and the fused single-step joint+decision head.
    static let requiredModelFiles: [String] = [
        "Preprocessor.mlmodelc",
        "Encoder.mlmodelc",
        "Decoder.mlmodelc",
        "JointDecision.mlmodelc",
    ]

    /// Token-id → sentencepiece piece map in FluidAudio's dict format.
    static let vocabularyFileName = "parakeet_vocab.json"

    /// `<Application Support>/FluidAudio/Models/omi-med-stt-v1-coreml`.
    public nonisolated static func modelDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Application Support", isDirectory: true)
        return appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(folderName, isDirectory: true)
    }

    public nonisolated static func isInstalled() -> Bool {
        isInstalled(at: modelDirectory())
    }

    /// Directory-parameterized core of ``isInstalled()`` so tests can exercise
    /// the required-file check against a temp dir.
    nonisolated static func isInstalled(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let allModelsPresent = requiredModelFiles.allSatisfy {
            fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        let vocabPresent = fileManager.fileExists(
            atPath: directory.appendingPathComponent(vocabularyFileName).path)
        return allModelsPresent && vocabPresent
    }

    /// Removes the installed bundle. Returns `true` only when the directory
    /// existed and is gone afterward; a no-op `false` when nothing was
    /// installed. The bundle re-downloads on next use.
    @discardableResult
    public nonisolated static func deleteModel() -> Bool {
        let fileManager = FileManager.default
        let directory = modelDirectory()
        guard fileManager.fileExists(atPath: directory.path) else { return false }
        do {
            try fileManager.removeItem(at: directory)
        } catch {
            return false
        }
        return !fileManager.fileExists(atPath: directory.path)
    }

    // MARK: - Download

    /// One entry in the HuggingFace `tree` manifest.
    struct ManifestEntry: Decodable, Sendable {
        struct LFS: Decodable, Sendable {
            /// sha256 of the file content for LFS-tracked files.
            let oid: String
        }

        let type: String
        let path: String
        let size: Int64?
        let lfs: LFS?
    }

    /// Files the bundle actually needs: the component `.mlmodelc` trees and
    /// the vocabulary. Repo housekeeping (README, .gitattributes) is skipped.
    nonisolated static func bundleFiles(in manifest: [ManifestEntry]) -> [ManifestEntry] {
        manifest.filter { entry in
            guard entry.type == "file" else { return false }
            if entry.path == vocabularyFileName { return true }
            return requiredModelFiles.contains { entry.path.hasPrefix($0 + "/") }
        }
    }

    /// Downloads the compiled bundle from ``huggingFaceRepo`` into
    /// ``modelDirectory()``. A cheap validate-and-return when already
    /// installed; otherwise fetches the repo manifest, downloads every bundle
    /// file into a staging directory (verifying the sha256 HuggingFace
    /// records for LFS-tracked files), and moves the staged tree into place
    /// so a torn download can never masquerade as an install.
    public nonisolated static func downloadModel(
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws {
        if isInstalled() { return }

        onProgress?("Fetching \(ParakeetModelVariant.omiMedV1.modelName) file list...")
        let manifest: [ManifestEntry]
        do {
            let manifestURL = URL(
                string: "https://huggingface.co/api/models/\(huggingFaceRepo)/tree/main?recursive=true")!
            let (data, response) = try await URLSession.shared.data(from: manifestURL)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw STTError.modelDownloadFailed
            }
            manifest = try JSONDecoder().decode([ManifestEntry].self, from: data)
        } catch let error as STTError {
            throw error
        } catch {
            throw STTError.modelDownloadFailed
        }

        let files = bundleFiles(in: manifest)
        let totalBytes = files.reduce(Int64(0)) { $0 + ($1.size ?? 0) }
        guard !files.isEmpty, totalBytes > 0 else {
            throw STTError.modelDownloadFailed
        }

        let fileManager = FileManager.default
        let stagingDir = modelDirectory()
            .deletingLastPathComponent()
            .appendingPathComponent(folderName + ".downloading", isDirectory: true)
        try? fileManager.removeItem(at: stagingDir)
        try fileManager.createDirectory(at: stagingDir, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: stagingDir) }

        var downloadedBytes: Int64 = 0
        for file in files {
            let destination = stagingDir.appendingPathComponent(file.path)
            try fileManager.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

            let source = URL(string: "https://huggingface.co/\(huggingFaceRepo)/resolve/main/\(file.path)")!
            let bytesBeforeFile = downloadedBytes
            try await downloadFile(
                from: source,
                to: destination,
                expectedSHA256: file.lfs?.oid,
                progress: { fileBytes in
                    let percent = Int((Double(bytesBeforeFile + fileBytes) / Double(totalBytes)) * 100)
                    onProgress?(
                        "Downloading \(ParakeetModelVariant.omiMedV1.modelName)... \(min(percent, 99))%")
                }
            )
            downloadedBytes += file.size ?? 0
        }

        guard isInstalled(at: stagingDir) else {
            throw STTError.modelDownloadFailed
        }

        // Replace-then-move keeps the real directory either absent or complete.
        try? fileManager.removeItem(at: modelDirectory())
        try fileManager.createDirectory(
            at: modelDirectory().deletingLastPathComponent(), withIntermediateDirectories: true)
        try fileManager.moveItem(at: stagingDir, to: modelDirectory())
        onProgress?("\(ParakeetModelVariant.omiMedV1.modelName) downloaded")
    }

    /// Downloads one file via a delegate session (the same pattern FluidAudio's
    /// `DownloadUtils` uses: OS-level file download with `didWriteData` byte
    /// progress), verifies the sha256 HuggingFace records for LFS-tracked files
    /// (which covers all the weights), and moves it into place.
    private nonisolated static func downloadFile(
        from source: URL,
        to destination: URL,
        expectedSHA256: String?,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws {
        do {
            let delegate = ByteProgressDelegate(onProgress: progress)
            // Dedicated session per file so delegate callbacks can't cross-talk.
            let session = URLSession(
                configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
            defer { session.finishTasksAndInvalidate() }

            let (tempURL, response) = try await session.download(from: source)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw STTError.modelDownloadFailed
            }

            if let expectedSHA256 {
                let digest = try sha256Hex(of: tempURL)
                guard digest == expectedSHA256.lowercased() else {
                    throw STTError.modelDownloadFailed
                }
            }

            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: tempURL, to: destination)
        } catch let error as STTError {
            throw error
        } catch {
            throw STTError.modelDownloadFailed
        }
    }

    /// Streaming sha256 of a file (8 MiB chunks — the encoder weights are
    /// ~1.1 GB, far too large for a single `Data(contentsOf:)`).
    private nonisolated static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Load

    /// Loads the installed bundle into a FluidAudio `AsrModels` value the
    /// shared TDT `AsrManager` accepts. Compute-unit placement mirrors
    /// `AsrModels.load` for v2: preprocessor pinned to CPU (its ops map to CPU
    /// anyway), everything else on CPU+ANE.
    static func load() async throws -> AsrModels {
        let directory = modelDirectory()
        guard isInstalled(at: directory) else {
            throw STTError.modelNotInstalled(
                "Omi Med STT v1 is not downloaded. Run `macparakeet-cli models download "
                    + "parakeet-omi-med-v1` or select the build in Settings to fetch it from "
                    + "huggingface.co/\(huggingFaceRepo)."
            )
        }

        let neuralEngineConfig = AsrModels.defaultConfiguration()
        let cpuOnlyConfig = MLModelConfiguration()
        cpuOnlyConfig.computeUnits = .cpuOnly

        do {
            let preprocessor = try await MLModel.load(
                contentsOf: directory.appendingPathComponent("Preprocessor.mlmodelc"),
                configuration: cpuOnlyConfig
            )
            let encoder = try await MLModel.load(
                contentsOf: directory.appendingPathComponent("Encoder.mlmodelc"),
                configuration: neuralEngineConfig
            )
            let decoder = try await MLModel.load(
                contentsOf: directory.appendingPathComponent("Decoder.mlmodelc"),
                configuration: neuralEngineConfig
            )
            let joint = try await MLModel.load(
                contentsOf: directory.appendingPathComponent("JointDecision.mlmodelc"),
                configuration: neuralEngineConfig
            )
            let vocabulary = try loadVocabulary(
                from: directory.appendingPathComponent(vocabularyFileName))

            return AsrModels(
                encoder: encoder,
                preprocessor: preprocessor,
                decoder: decoder,
                joint: joint,
                configuration: neuralEngineConfig,
                vocabulary: vocabulary,
                version: .v2
            )
        } catch let error as STTError {
            throw error
        } catch {
            throw STTError.engineStartFailed(
                "Failed to load Omi Med STT v1 from \(directory.path): \(error.localizedDescription)"
            )
        }
    }

    /// Byte-progress relay for `URLSession.download(from:)` — mirrors
    /// FluidAudio's `DownloadProgressDelegate` pattern.
    private final class ByteProgressDelegate: NSObject, URLSessionDownloadDelegate, Sendable {
        private let onProgress: @Sendable (Int64) -> Void

        init(onProgress: @escaping @Sendable (Int64) -> Void) {
            self.onProgress = onProgress
        }

        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didWriteData bytesWritten: Int64,
            totalBytesWritten: Int64,
            totalBytesExpectedToWrite: Int64
        ) {
            onProgress(totalBytesWritten)
        }

        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didFinishDownloadingTo location: URL
        ) {
            // Required by the protocol — the async download(from:) API owns the file.
        }
    }

    /// Parses FluidAudio's dict-format vocabulary (`{"<token_id>": "<piece>"}`).
    nonisolated static func loadVocabulary(from url: URL) throws -> [Int: String] {
        let data = try Data(contentsOf: url)
        guard let entries = try JSONSerialization.jsonObject(with: data) as? [String: String] else {
            throw STTError.engineStartFailed(
                "Omi Med vocabulary at \(url.path) is not a {\"id\": \"token\"} JSON dictionary."
            )
        }
        var vocabulary: [Int: String] = [:]
        vocabulary.reserveCapacity(entries.count)
        for (key, value) in entries {
            guard let tokenId = Int(key) else { continue }
            vocabulary[tokenId] = value
        }
        guard !vocabulary.isEmpty else {
            throw STTError.engineStartFailed("Omi Med vocabulary at \(url.path) is empty.")
        }
        return vocabulary
    }
}
