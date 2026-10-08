import AppKit
import Darwin
import UniformTypeIdentifiers

func logNative(_ message: String, level: String = "INFO") {
    let time = DateFormatter()
    time.locale = Locale(identifier: "en_US_POSIX")
    time.dateFormat = "HH:mm:ss"
    fputs("\(time.string(from: Date())) \(level) macos: \(message)\n", stderr)
    fflush(stderr)
}

let repoRoot: URL = {
    guard let resource = Bundle.main.url(forResource: "RepositoryPath", withExtension: "txt"),
          let path = try? String(contentsOf: resource, encoding: .utf8),
          path.hasPrefix("/") else {
        fatalError("[Resonance] Missing repository path; rebuild with ./r build-macos")
    }
    return URL(fileURLWithPath: path, isDirectory: true)
}()

func readEnvValue(_ key: String, fallback: String) -> String {
    let envFile = repoRoot.appendingPathComponent(".env")
    guard let content = try? String(contentsOf: envFile, encoding: .utf8) else { return fallback }
    for line in content.components(separatedBy: .newlines) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.hasPrefix("#") else { continue }
        let parts = trimmed.components(separatedBy: "=")
        guard parts.count >= 2 else { continue }
        guard parts[0].trimmingCharacters(in: .whitespaces) == key else { continue }

        var val = parts.dropFirst().joined(separator: "=").trimmingCharacters(in: .whitespaces)
        if let commentIndex = val.firstIndex(of: "#") {
            val = String(val[..<commentIndex]).trimmingCharacters(in: .whitespaces)
        }
        val = val.replacingOccurrences(of: "\"", with: "").replacingOccurrences(of: "'", with: "")
        return val.isEmpty ? fallback : val
    }
    return fallback
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var backend: Process?
    private var backendLogURL: URL?
    var port = "8000"
    var isQuitting = false
    var captureEngine: CaptureEngine?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Single instance guard
        let bundleID = Bundle.main.bundleIdentifier ?? "com.resonance.app"
        if NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).count > 1 {
            NSApp.terminate(nil)
            return
        }

        port = readEnvValue("RESONANCE_PORT", fallback: "8000")

        startBackend()
        setupStatusItem()

        // Initialize native audio capture and IPC.
        captureEngine = CaptureEngine()
    }

    private func createMenuItem(title: String, action: Selector, key: String, symbolName: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
        if let symbolImage = NSImage(systemSymbolName: symbolName, accessibilityDescription: title) {
            item.image = symbolImage
        }
        return item
    }

    func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            guard let image = NSImage(named: "StatusBarIcon") else {
                fatalError("[Resonance] Missing required bundle resource: StatusBarIcon")
            }
            image.isTemplate = true
            button.image = image
        }

        let menu = NSMenu()
        menu.addItem(createMenuItem(title: "Open in Browser", action: #selector(openBrowser), key: "o", symbolName: "globe"))
        menu.addItem(NSMenuItem.separator())

        menu.addItem(createMenuItem(title: "Settings", action: #selector(openSettings), key: ",", symbolName: "gearshape"))
        menu.addItem(createMenuItem(title: "Save Diagnostics…", action: #selector(saveDiagnostics(_:)), key: "", symbolName: "square.and.arrow.down"))

        menu.addItem(NSMenuItem.separator())
        menu.addItem(createMenuItem(title: "Quit", action: #selector(quit), key: "q", symbolName: "power"))

        statusItem.menu = menu
    }

    func startBackend() {
        // Launch the checkout backend with built web assets; no Node runtime is needed.
        backend = Process()
        backend?.executableURL = URL(fileURLWithPath: "/bin/zsh")
        backend?.arguments = ["-l", "-c", "export PATH=\"/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH\"; exec ./r serve-local"]
        backend?.currentDirectoryURL = repoRoot
        var environment = ProcessInfo.processInfo.environment
        // Swift owns the complete backend log; avoid a second Python file logger.
        // Empty values also prevent dotenv from restoring file logging from .env.
        environment["RESONANCE_LOG_TO_FILE"] = "0"
        environment["RESONANCE_LOG_FILE"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        backend?.environment = environment

        do {
            let logURL = try createBackendLog()
            let output = try FileHandle(forWritingTo: logURL)
            defer { try? output.close() }
            _ = try output.seekToEnd()
            backendLogURL = logURL
            // Include native capture diagnostics in the same launch log.
            fflush(stdout)
            fflush(stderr)
            guard dup2(output.fileDescriptor, STDOUT_FILENO) != -1,
                  dup2(output.fileDescriptor, STDERR_FILENO) != -1 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            let revisionURL = Bundle.main.url(forResource: "BuildRevision", withExtension: "txt")
            let revision = revisionURL.flatMap { try? String(contentsOf: $0, encoding: .utf8) } ?? "unavailable"
            let started = ISO8601DateFormatter()
            started.timeZone = .current
            fputs("# Native session: pid=\(getpid()); started=\(started.string(from: Date())); app_revision=\(revision)\n", stderr)
            fflush(stderr)
            // Model downloads write progress to stderr. An unread pipe can fill
            // and block the backend; inherited stdout can outlive its reader.
            backend?.standardOutput = output
            backend?.standardError = output

            backend?.terminationHandler = { [weak self] process in
                guard process.terminationStatus != 0 else { return }
                DispatchQueue.main.async {
                    guard let self = self, self.backend === process, !self.isQuitting else { return }
                    self.showBackendFailure(
                        title: "Resonance Service Stopped",
                        details: "Backend exited with code \(process.terminationStatus), reason \(process.terminationReason.rawValue)."
                    )
                }
            }
            try backend?.run()
        } catch {
            showBackendFailure(title: "Backend Failed to Start", details: String(describing: error))
        }
    }

    private func showBackendFailure(title: String, details: String) {
        logNative("\(title): \(details)", level: "ERROR")
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = backendLogURL == nil ? details :
            "Save diagnostics and send the ZIP file with a short description of what went wrong. To restart Resonance, choose Quit from the menu, then open the app again."
        alert.alertStyle = .critical
        if backendLogURL != nil { alert.addButton(withTitle: "Save Diagnostics…") }
        alert.addButton(withTitle: "Close")
        NSApp.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn, backendLogURL != nil {
            saveDiagnostics(nil)
        }
        // Keep the menu available for diagnostics and quitting after a startup failure.
    }

    @objc func openBrowser() {
        guard let url = URL(string: "http://localhost:\(port)") else { return }
        NSWorkspace.shared.open(url)
    }

    @objc func openSettings() {
        let envPath = repoRoot.appendingPathComponent(".env").path
        let task = Process()
        task.launchPath = "/usr/bin/open"
        task.arguments = ["-e", envPath] // -e forces opening in TextEdit
        try? task.run()
    }

    private func diagnosticsTool(_ arguments: [String]) -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-l", "-c", "export PATH=\"/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH\"; exec ./r diagnostics \"$@\"", "resonance-diagnostics"] + arguments
        process.currentDirectoryURL = repoRoot
        process.standardInput = FileHandle.nullDevice
        return process
    }

    private func createBackendLog() throws -> URL {
        let process = diagnosticsTool(["--prepare-log"])
        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        try process.run()
        let data = try output.fileHandleForReading.readToEnd() ?? Data()
        process.waitUntilExit()
        let details = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let path = details.components(separatedBy: .newlines).last ?? ""
        guard process.terminationStatus == 0, path.hasPrefix("/") else {
            throw NSError(domain: "Resonance", code: Int(process.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: "Could not prepare the session log.\n\(details.suffix(32_768))"])
        }
        return URL(fileURLWithPath: path)
    }

    @objc func saveDiagnostics(_ sender: NSMenuItem?) {
        guard let log = backendLogURL else { return }
        NSApp.activate(ignoringOtherApps: true)
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.zip]
        panel.nameFieldStringValue = "Resonance Diagnostics.zip"
        panel.message = "Send the saved ZIP file with what you clicked, what you expected, and what happened."
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        let process = diagnosticsTool(["--log-file", log.path, "--output", destination.path])
        sender?.isEnabled = false
        DispatchQueue.global(qos: .utility).async {
            var failure: String?
            do {
                try process.run()
                process.waitUntilExit()
                if process.terminationStatus != 0 {
                    failure = "Could not save diagnostics. Check the current session log for details."
                }
            } catch {
                failure = error.localizedDescription
            }
            let message = failure
            DispatchQueue.main.async {
                sender?.isEnabled = true
                if let message {
                    let alert = NSAlert()
                    alert.messageText = "Diagnostics Export Failed"
                    alert.informativeText = message
                    alert.runModal()
                } else {
                    NSWorkspace.shared.activateFileViewerSelecting([destination])
                }
            }
        }
    }

    private func terminateProcessGroup(_ process: Process) {
        let pid = process.processIdentifier
        guard pid > 0 else { return }

        // Invariants: Foundation.Process places the child in an isolated process group (PGID == PID).
        // Signaling the negative PGID ensures intermediate shells and spawned Python workers
        // are torn down reliably without abandoning orphaned processes that hold network ports.
        kill(-pid, SIGTERM)
        kill(pid, SIGTERM)

        for _ in 0..<15 {
            if !process.isRunning { return }
            Thread.sleep(forTimeInterval: 0.1)
        }

        if process.isRunning {
            kill(-pid, SIGKILL)
            kill(pid, SIGKILL)
        }
    }

    @objc func quit() {
        NSApp.terminate(nil)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let process = backend, process.isRunning else {
            return .terminateNow
        }

        isQuitting = true

        DispatchQueue.global().async {
            self.terminateProcessGroup(process)

            DispatchQueue.main.async {
                NSApp.reply(toApplicationShouldTerminate: true)
            }
        }

        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {}
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
