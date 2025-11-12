/* Avatar upload helper for profile settings */
(function () {
    const form = document.querySelector("[data-avatar-form]");
    if (!form) {
        return;
    }

    const fileInput = form.querySelector("[data-avatar-input]");
    if (!fileInput) {
        return;
    }

    const fileTrigger = form.querySelector("[data-avatar-trigger]");
    const fileLabel = form.querySelector("[data-avatar-label]");
    const fileMeta = form.querySelector("[data-avatar-filemeta]");
    const feedback = form.querySelector("[data-avatar-feedback]");
    const submitButton = form.querySelector("[data-avatar-submit]");
    const submitLabel = form.querySelector("[data-avatar-submit-label]");
    const previewImage = document.querySelector(".profile-avatar-preview img");

    const defaultLabelText = fileLabel ? fileLabel.textContent.trim() : "";
    const defaultMetaText = fileMeta ? fileMeta.textContent.trim() : "";
    const defaultFeedbackText = feedback ? feedback.textContent.trim() : "";
    const defaultPreviewSrc = previewImage ? previewImage.getAttribute("src") : null;

    const allowedMimeTypes = new Set([
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/gif",
        "image/webp"
    ]);
    const allowedExtensions = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp"]);
    const maxBytes = 5 * 1024 * 1024;

    function resetVisualState() {
        if (fileTrigger) {
            fileTrigger.classList.remove("is-invalid", "is-valid");
        }
        if (feedback) {
            feedback.textContent = defaultFeedbackText;
            feedback.classList.remove("is-error", "is-success");
        }
        if (fileMeta) {
            fileMeta.textContent = defaultMetaText;
        }
        if (fileLabel) {
            fileLabel.textContent = defaultLabelText || "Select photo";
        }
    }

    function formatBytes(bytes) {
        if (!bytes) {
            return "0 B";
        }
        const units = ["B", "KB", "MB"];
        const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        const value = bytes / Math.pow(1024, idx);
        const decimals = idx === 0 ? 0 : 1;
        return value.toFixed(decimals) + " " + units[idx];
    }

    function getExtension(fileName) {
        if (!fileName || fileName.indexOf(".") === -1) {
            return "";
        }
        return "." + fileName.split(".").pop().toLowerCase();
    }

    function validateFile(file) {
        if (!file) {
            return { valid: false, error: "Select an image before uploading." };
        }

        const mimeType = (file.type || "").toLowerCase();
        const extension = getExtension(file.name);
        const isAllowedType = allowedMimeTypes.has(mimeType) || allowedExtensions.has(extension);

        if (!isAllowedType) {
            return { valid: false, error: "Please choose a PNG, JPG, GIF, or WebP image." };
        }

        if (file.size > maxBytes) {
            return { valid: false, error: "That file is too large. The limit is 5 MB." };
        }

        return { valid: true };
    }

    function setError(message) {
        if (feedback) {
            feedback.textContent = message;
            feedback.classList.add("is-error");
            feedback.classList.remove("is-success");
        }
        if (fileTrigger) {
            fileTrigger.classList.remove("is-valid");
            fileTrigger.classList.add("is-invalid");
        }
        if (fileMeta) {
            fileMeta.textContent = "";
        }
        if (fileLabel && defaultLabelText) {
            fileLabel.textContent = defaultLabelText;
        }
        if (previewImage && defaultPreviewSrc) {
            previewImage.src = defaultPreviewSrc;
        }
        fileInput.value = "";
    }

    function setSuccess(file) {
        if (fileTrigger) {
            fileTrigger.classList.remove("is-invalid");
            fileTrigger.classList.add("is-valid");
        }
        if (fileLabel) {
            fileLabel.textContent = "Change photo";
        }
        if (fileMeta) {
            fileMeta.textContent = file.name + " • " + formatBytes(file.size);
        }
        if (feedback) {
            feedback.textContent = "Looks good! Click Upload to save your new photo.";
            feedback.classList.add("is-success");
            feedback.classList.remove("is-error");
        }
        if (previewImage && typeof FileReader !== "undefined") {
            const reader = new FileReader();
            reader.addEventListener("load", function (event) {
                if (typeof event.target.result === "string") {
                    previewImage.src = event.target.result;
                }
            });
            reader.readAsDataURL(file);
        }
    }

    function clearLoadingState() {
        if (submitButton) {
            submitButton.classList.remove("is-loading");
            submitButton.removeAttribute("aria-busy");
            submitButton.disabled = false;
        }
        if (submitLabel) {
            submitLabel.textContent = "Upload";
        }
    }

    function applyLoadingState() {
        if (submitButton) {
            submitButton.classList.add("is-loading");
            submitButton.setAttribute("aria-busy", "true");
            submitButton.disabled = true;
        }
        if (submitLabel) {
            submitLabel.textContent = "Uploading...";
        }
    }

    fileInput.addEventListener("change", function () {
        resetVisualState();

        const file = fileInput.files && fileInput.files[0];
        if (!file) {
            if (previewImage && defaultPreviewSrc) {
                previewImage.src = defaultPreviewSrc;
            }
            return;
        }

        const validation = validateFile(file);
        if (!validation.valid) {
            setError(validation.error);
            if (fileInput instanceof HTMLElement) {
                fileInput.focus();
            }
            return;
        }

        setSuccess(file);
    });

    form.addEventListener("submit", function (event) {
        const file = fileInput.files && fileInput.files[0];
        const validation = validateFile(file);

        if (!validation.valid) {
            event.preventDefault();
            setError(validation.error);
            if (fileInput instanceof HTMLElement) {
                fileInput.focus();
            }
            clearLoadingState();
            return;
        }

        applyLoadingState();
    });

    form.addEventListener("reset", function () {
        resetVisualState();
        if (previewImage && defaultPreviewSrc) {
            previewImage.src = defaultPreviewSrc;
        }
        clearLoadingState();
    });
})();

