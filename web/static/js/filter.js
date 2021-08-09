function filter_by(query, metric_name) {
  $(`#finishedexperiments .table__drow`).filter(function () {
    $(this).toggle(
      $(`.table__dcell--${metric_name} p`, this).text().indexOf(query) > -1
    )
  })
}

$(document).ready(function (){
  console.log("ready");
  $(".filter_box--name").on(
    "keyup",
    function () {filter_by($(this).val(), "username");},
  )
  $(window).on("pageshow", function (){
    filter_by($(".filter_box--name").val(), "username")
  })
})

