$(document).ready(function () {
    function handleBitcoinResponse(response, status, xhr){
        //console.log(response, status, xhr)
        console.log(response)
        let bpi = response.bpi
        $("#BitResults").empty();
        //$("#BitResults").text(JSON.stringify(bpi));

        for(let i in bpi) {
            console.log(i+ " " +bpi[i])
            $("#BitResults").append(i+ " " +bpi[i]+ "<br>")
        };
        renderChart(bpi);
    };

    function bitconButtonClicked(event) {
        var url = " https://api.coindesk.com/v1/bpi/currentprice.json";
        var url2 = "https://api.coindesk.com/v1/bpi/historical/close.json?start=2018-03-01"

        $.getJSON(url2, handleBitcoinResponse);
    };

    $("#BitCoinButton").click(bitconButtonClicked);

    function renderChart(bpi) {
        Highcharts.chart('container', {
            chart: {
                type: 'line'
            },
            title: {
                text: '30 days Bitcoin Average Price'
            },
            subtitle: {
                text: 'Source: butaire.com'
            },
            xAxis: {
                categories: Object.keys(bpi)
            },
            yAxis: {
                title: {
                    text: 'Price in USD'
                }
            },
            plotOptions: {
                line: {
                    dataLabels: {
                        enabled: true
                    },
                    enableMouseTracking: true
                }
            },
            series: [{
                name: 'Btcoin',
                data: Object.values(bpi)
            }, {
                name: 'Baseline',
                data: [3.9, 4.2, 5.7, 8.5, 11.9, 15.2, 17.0, 16.6, 14.2, 10.3, 6.6, 4.8]
            }]
        });
    }

})