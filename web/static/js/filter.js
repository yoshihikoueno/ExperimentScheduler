class KeywordFilter {
  constructor(container, text_extractor) {
    this.filter_results_permetric = {}
    this.filter_results = []
    this.container = container
    this.text_extractor = text_extractor
  }
  filter(query, metric) {
    const that = this
    this.filter_results_permetric[metric] = $(this.container).map(function (){ return that.text_extractor(metric, this).indexOf(query) > -1 })
    this.filter_results = Object.values(this.filter_results_permetric)[0].map(
      (i, e) => Object.keys(this.filter_results_permetric).map(metric => this.filter_results_permetric[metric][i]).reduce((curr, acc) => curr && acc)
    )
    this.apply()
  }
  apply() {
    const that = this
    $(this.container).map(function(i) {$(this).toggle(that.filter_results[i])})
  }
}

$(document).ready(function (){
  keyword_filter = new KeywordFilter(
    `#finishedexperiments .table__drow`,
    (metric, context) => $(`.table__dcell--${metric} p`, context).text(),
  )
  $(".filter_box--name").on(
    "keyup",
    function () {keyword_filter.filter($(this).val(), "username")},
  )
  $(".filter_box--experiment").on(
    "keyup",
    function () {keyword_filter.filter($(this).val(), "experiment_name")},
  )
  $(window).on("pageshow", function (){
    keyword_filter.filter($(".filter_box--name").val(), "username")
    keyword_filter.filter($(".filter_box--experiment").val(), "experiment_name")
  })
})

