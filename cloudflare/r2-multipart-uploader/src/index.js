const json = (value, status = 200) =>
  Response.json(value, {
    status,
    headers: {
      "cache-control": "no-store",
    },
  });

const validKey = (key) =>
  typeof key === "string" &&
  key.startsWith("datasets/") &&
  key.length <= 1024 &&
  !key.includes("..");

const authenticated = (request, env) => {
  if (!env.UPLOAD_TOKEN) {
    return false;
  }
  return request.headers.get("authorization") === `Bearer ${env.UPLOAD_TOKEN}`;
};

const parseJson = async (request) => {
  try {
    return await request.json();
  } catch {
    return null;
  }
};

export default {
  async fetch(request, env) {
    if (!authenticated(request, env)) {
      return json({ error: "unauthorized" }, 401);
    }

    const url = new URL(request.url);
    try {
      if (request.method === "POST" && url.pathname === "/multipart/create") {
        const body = await parseJson(request);
        if (!body || !validKey(body.key)) {
          return json({ error: "invalid key" }, 400);
        }
        const upload = await env.ASSETS.createMultipartUpload(body.key, {
          httpMetadata: {
            contentType: body.contentType || "application/octet-stream",
            contentDisposition: body.contentDisposition || "attachment",
            cacheControl:
              body.cacheControl ||
              "public, max-age=31536000, immutable",
          },
          customMetadata: {
            sha256: body.sha256 || "",
          },
        });
        return json({ key: upload.key, uploadId: upload.uploadId });
      }

      if (request.method === "PUT" && url.pathname === "/multipart/part") {
        const key = url.searchParams.get("key");
        const uploadId = url.searchParams.get("uploadId");
        const partNumber = Number(url.searchParams.get("partNumber"));
        const contentLength = Number(request.headers.get("content-length"));
        if (
          !validKey(key) ||
          !uploadId ||
          !Number.isInteger(partNumber) ||
          partNumber < 1 ||
          partNumber > 10000 ||
          !Number.isFinite(contentLength) ||
          contentLength <= 0 ||
          contentLength > 96 * 1024 * 1024 ||
          !request.body
        ) {
          return json({ error: "invalid part request" }, 400);
        }
        const upload = env.ASSETS.resumeMultipartUpload(key, uploadId);
        const part = await upload.uploadPart(partNumber, request.body);
        return json({
          partNumber: part.partNumber,
          etag: part.etag,
        });
      }

      if (
        request.method === "POST" &&
        url.pathname === "/multipart/complete"
      ) {
        const body = await parseJson(request);
        if (
          !body ||
          !validKey(body.key) ||
          !body.uploadId ||
          !Array.isArray(body.parts) ||
          body.parts.length === 0
        ) {
          return json({ error: "invalid completion request" }, 400);
        }
        const upload = env.ASSETS.resumeMultipartUpload(
          body.key,
          body.uploadId,
        );
        const object = await upload.complete(body.parts);
        return json({
          key: object.key,
          size: object.size,
          etag: object.etag,
        });
      }

      if (request.method === "POST" && url.pathname === "/multipart/abort") {
        const body = await parseJson(request);
        if (!body || !validKey(body.key) || !body.uploadId) {
          return json({ error: "invalid abort request" }, 400);
        }
        const upload = env.ASSETS.resumeMultipartUpload(
          body.key,
          body.uploadId,
        );
        await upload.abort();
        return json({ aborted: true });
      }

      return json({ error: "not found" }, 404);
    } catch (error) {
      return json(
        {
          error: "r2 operation failed",
          detail: error instanceof Error ? error.message : String(error),
        },
        500,
      );
    }
  },
};
