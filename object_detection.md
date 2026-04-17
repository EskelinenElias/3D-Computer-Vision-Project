cv2.inRange
mask = cv2.inRange(hsv_image, low, high) — for every pixel, returns 255 if all three HSV channels are within [low, high] (inclusive), else 0. So mask is a uint8 binary image the same size as the input.

HSV is used instead of RGB because hue (the first channel) encodes color independently of brightness — so a red cube in shadow and a red cube in direct light both have H≈0, while their RGB values would differ wildly. We keep the saturation (≥60–100) and value (≥40–80) lower bounds to reject grey/white/black pixels that have no meaningful hue.

Gotcha: hue is cyclic on [0, 179] in OpenCV (8-bit, so the usual 0–360° is halved). Red sits at both ends of the scale, which is why object_detection.py:5-6 uses two ranges (0–10 and 165–179) and ORs the masks together:


mask |= cv2.inRange(hsv_image, np.array(low), np.array(high))
cv2.morphologyEx — open + close cleanup
After inRange you get a noisy mask: scattered "salt" pixels where stray lighting briefly matches the hue, plus small "pepper" holes inside the real blob. Morphological operations are two-step filters based on a structuring element (here a 5×5 ellipse at object_detection.py:26).

The two primitives:

Erosion — a pixel stays white only if every pixel under the structuring element is white. Shrinks blobs, kills isolated white noise.
Dilation — a pixel becomes white if any pixel under the element is white. Grows blobs, fills small holes.
morphologyEx is a convenience wrapper that composes them:

MORPH_OPEN = erode then dilate. Removes small white specks (tiny noise) without shrinking the real blob — because the initial erosion kills any blob smaller than the kernel, then the dilation re-grows what survived back to its original size.
MORPH_CLOSE = dilate then erode. Fills small black holes inside blobs (e.g. a glare spot in a red cube that misses the HSV range) without growing the outer boundary.
So object_detection.py:27-28 applies both: first open (despeckle), then close (fill pinholes). The order matters — opening first avoids closing noise specks into solid shapes.

Kernel size tradeoff: 5×5 is small enough to preserve fine features like the target ring's inner hole, but large enough to clean up reasonable noise. If you used 15×15 you'd likely close the ring and accidentally turn targets into solid blobs, breaking the hole-based target/cube distinction.

cv2.findContours + hierarchy

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
Walks the binary image and traces the boundaries between white and black regions. Two parameters matter:

Retrieval mode RETR_TREE — returns all contours (outer boundaries and inner holes), and records their parent/child tree. Alternatives: RETR_EXTERNAL (outer only, cheaper) or RETR_LIST (all, but forgets the tree). We specifically want the tree — that's how we distinguish solid cubes from ring targets.
Approximation CHAIN_APPROX_SIMPLE — for a straight edge, only store the endpoints instead of every pixel along it. Huge memory win for blocky shapes, no loss for our purposes.
The hierarchy array
hierarchy has shape (1, N, 4) where each row is [next_sibling, prev_sibling, first_child, parent] — each entry is either an index into contours or -1 if that relationship doesn't exist.

The check at object_detection.py:47:


has_child = hierarchy[0][i][2] != -1
reads: "does contour i have a first child?" A child contour exists when there's a hole inside the current contour — i.e. a black region completely surrounded by white.

Mapping that to physical objects:

Solid cube top face → single closed contour, no child → has_child = False → matches has_hole=False
Target ring → outer ring boundary is one contour; the inner hole edge is a child of that contour → has_child = True → matches has_hole=True
This is why the morphology step is so important: if you over-close, you'd fill the ring's hole, destroying the child contour, and targets would start being misclassified as cubes. If you under-clean, stray noise contours add spurious children to real blobs. The 5×5 elliptical kernel at object_detection.py:26 is a sweet spot that's been tuned for this specific scene.

Area filter
object_detection.py:45 discards any contour with cv2.contourArea < 500 pixels. Final safety net — even if morphology leaves a small rogue contour, it won't be returned. Then the list is sorted largest-first so callers can just take blobs[0] as "the main blob."