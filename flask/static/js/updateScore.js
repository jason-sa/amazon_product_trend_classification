function updateScore(){
  const theReview = document.getElementById('review_text').value;

  // Fill in the table
  const originalComment = document.getElementById('original_comment')
  const impComment = document.getElementById('improved_comment');
  const origCommentProb = document.getElementById('original_prob');
  const impCommentProb = document.getElementById('improved_prob');

  if (theReview.length < 5) {
      outputElement.textContent = 'Need more text to provide review';
      return;
  }

  $.ajax({
    type: 'POST',
    contentType: "application/json; charset=utf-8",
    url: '/trend_score',
    async: true,
    data: JSON.stringify({
      review: theReview
    }),
    success: (response) => {
      originalComment.textContent = theReview
      impComment.innerHTML = response.best_comment;

      if (response.orig_score >= response.best_score){
        origCommentProb.innerHTML = '<span style="font-weight:bold">' + response.orig_score + '</span>';
        impCommentProb.textContent = response.best_score;
      }
      else{
        origCommentProb.textContent = response.orig_score;
        impCommentProb.innerHTML = '<span style="font-weight:bold">' + response.best_score + '</span>';

      }

      // smile(response.orig_score);
    },
    error: (response) => {
      outputElement.textContent = 'INVALID';
    }
  })
}
