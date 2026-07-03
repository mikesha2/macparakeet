import Foundation
import XCTest

@testable import MacParakeetCore

/// File-layout and vocabulary-parsing coverage for the local-install-only
/// Omi Med Parakeet bundle. Model loading itself needs the real ~1.1 GB
/// CoreML artifacts, so it is exercised via the CLI/dev-app, not unit tests.
final class OmiMedParakeetModelTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omi-med-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func installFixtureBundle(omitting omitted: Set<String> = []) throws {
        // The check only requires presence; compiled-model *content* is the
        // CoreML runtime's concern at load time.
        for name in OmiMedParakeetModel.requiredModelFiles where !omitted.contains(name) {
            try FileManager.default.createDirectory(
                at: tempDir.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        if !omitted.contains(OmiMedParakeetModel.vocabularyFileName) {
            try Data(#"{"0": "\#u{2581}the"}"#.utf8).write(
                to: tempDir.appendingPathComponent(OmiMedParakeetModel.vocabularyFileName))
        }
    }

    func testIsInstalledRequiresEveryComponentAndVocabulary() throws {
        XCTAssertFalse(OmiMedParakeetModel.isInstalled(at: tempDir))

        try installFixtureBundle()
        XCTAssertTrue(OmiMedParakeetModel.isInstalled(at: tempDir))
    }

    func testIsInstalledFailsWhenAnyComponentIsMissing() throws {
        for missing in OmiMedParakeetModel.requiredModelFiles + [OmiMedParakeetModel.vocabularyFileName] {
            let caseDir = tempDir.appendingPathComponent(missing, isDirectory: true)
                .appendingPathExtension("case")
            try FileManager.default.createDirectory(at: caseDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: caseDir) }

            for name in OmiMedParakeetModel.requiredModelFiles where name != missing {
                try FileManager.default.createDirectory(
                    at: caseDir.appendingPathComponent(name), withIntermediateDirectories: true)
            }
            if missing != OmiMedParakeetModel.vocabularyFileName {
                try Data("{}".utf8).write(
                    to: caseDir.appendingPathComponent(OmiMedParakeetModel.vocabularyFileName))
            }
            XCTAssertFalse(
                OmiMedParakeetModel.isInstalled(at: caseDir),
                "bundle missing \(missing) must not report installed"
            )
        }
    }

    func testLoadVocabularyParsesDictFormat() throws {
        let vocabURL = tempDir.appendingPathComponent("vocab.json")
        try Data(#"{"0": "\#u{2581}the", "1023": "z", "not-a-number": "x"}"#.utf8).write(to: vocabURL)

        let vocabulary = try OmiMedParakeetModel.loadVocabulary(from: vocabURL)
        XCTAssertEqual(vocabulary[0], "\u{2581}the")
        XCTAssertEqual(vocabulary[1023], "z")
        XCTAssertEqual(vocabulary.count, 2, "non-integer keys are dropped")
    }

    func testLoadVocabularyRejectsNonDictAndEmptyFiles() throws {
        let arrayURL = tempDir.appendingPathComponent("array.json")
        try Data(#"["a", "b"]"#.utf8).write(to: arrayURL)
        XCTAssertThrowsError(try OmiMedParakeetModel.loadVocabulary(from: arrayURL))

        let emptyURL = tempDir.appendingPathComponent("empty.json")
        try Data("{}".utf8).write(to: emptyURL)
        XCTAssertThrowsError(try OmiMedParakeetModel.loadVocabulary(from: emptyURL))
    }

    func testModelDirectoryLivesUnderFluidAudioModels() {
        let path = OmiMedParakeetModel.modelDirectory().path
        XCTAssertTrue(path.hasSuffix("FluidAudio/Models/\(OmiMedParakeetModel.folderName)"))
    }

    func testBundleFilesSelectsComponentTreesAndVocabOnly() throws {
        // Mirrors the HuggingFace tree API shape. Repo housekeeping (README,
        // .gitattributes) and directory entries must not be downloaded.
        let manifestJSON = """
            [
              {"type": "file", "path": ".gitattributes", "size": 1519},
              {"type": "file", "path": "README.md", "size": 4872},
              {"type": "directory", "path": "Encoder.mlmodelc"},
              {"type": "file", "path": "Encoder.mlmodelc/weights/weight.bin", "size": 1198547712,
               "lfs": {"oid": "abc123"}},
              {"type": "file", "path": "Encoder.mlmodelc/model.mil", "size": 960148},
              {"type": "file", "path": "Decoder.mlmodelc/coremldata.bin", "size": 554,
               "lfs": {"oid": "def456"}},
              {"type": "file", "path": "parakeet_vocab.json", "size": 16543}
            ]
            """
        let manifest = try JSONDecoder().decode(
            [OmiMedParakeetModel.ManifestEntry].self, from: Data(manifestJSON.utf8))

        let files = OmiMedParakeetModel.bundleFiles(in: manifest)
        XCTAssertEqual(
            Set(files.map(\.path)),
            [
                "Encoder.mlmodelc/weights/weight.bin",
                "Encoder.mlmodelc/model.mil",
                "Decoder.mlmodelc/coremldata.bin",
                "parakeet_vocab.json",
            ]
        )
        // LFS sha256 digests ride along for download verification.
        XCTAssertEqual(
            files.first { $0.path == "Encoder.mlmodelc/weights/weight.bin" }?.lfs?.oid,
            "abc123"
        )
    }
}
