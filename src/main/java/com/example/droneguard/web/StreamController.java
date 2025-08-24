package com.example.droneguard.web;

import com.example.droneguard.video.VideoCaptureLoop;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.nio.charset.StandardCharsets;

@Controller
public class StreamController {
  private final VideoCaptureLoop video;

  public StreamController(VideoCaptureLoop video) {
    this.video = video;
  }

  @GetMapping(value = "/stream", produces = MediaType.MULTIPART_MIXED_VALUE)
  public ResponseEntity<StreamingResponseBody> stream() {
      final String boundary = "frame";
      return ResponseEntity.ok()
              .header("Content-Type", "multipart/x-mixed-replace; boundary=" + boundary)
              .body(outputStream -> {
                  while (true) {
                      byte[] jpg = video.getLatestJpeg();
                      if (jpg.length > 0) {
                          outputStream.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.US_ASCII));
                          outputStream.write(("Content-Type: image/jpeg\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
                          outputStream.write(jpg);
                          outputStream.write("\r\n".getBytes(StandardCharsets.US_ASCII));
                          outputStream.flush();
                      }
                      try {
                          Thread.sleep(10);
                      } catch (InterruptedException e) {
                          Thread.currentThread().interrupt();
                          break;
                      }
                  }
              });
  }

  @GetMapping("/")
  public String index() { return "redirect:/stream"; }
}
