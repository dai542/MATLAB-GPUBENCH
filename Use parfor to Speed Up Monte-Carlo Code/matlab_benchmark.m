function matlab_benchmark()
    nPlayers = 20;
    incr = 0.05;
    dropoutRate = 0.01;
    workers = [1 2 4 8 16 32];
    elapsedTimes = zeros(1,numel(workers));
    nTrials = 10000;

    parpool(32);

    results = zeros(numel(workers), nTrials); % to store result

    for k = 1:numel(workers)    
        t = zeros(1,5);
        B = zeros(1, nTrials);
        for j = 1:5
            tic
            parfor (i = 1:nTrials, workers(k))
                s = RandStream('Threefry');
                s.Substream = i;
                bids = dollarAuctionStream(nPlayers,incr,dropoutRate,s);
                B(i) = bids.Bid(end);
            end
            t(j) = toc;
        end
        
        results(k, :) = B % to store result
        elapsedTimes(k) = min(t);
    end

    disp(results);

    speedup = elapsedTimes(1) ./ elapsedTimes;
    plot(workers,speedup)
    xlabel('Number of workers')
    ylabel('Computational speedup')
    saveas(gcf, 'speedup_plot.png');
end

function [bids, dropouts] = dollarAuctionStream(nPlayers, incr, dropoutRate, s)
    % Simulate the auction of an item with value origValue using a
    % stochastic model.
    %
    % The function returns two outputs: bids and dropouts.
    % bids has a row for every bid placed in the auction.
    % dropouts has a row for each time a player drops out of the auction.
    
    players = 1:nPlayers;
    
    % Columns represent a player, bid, and epoch
    bids = zeros(0,3);
    
    % Columns represent a player and epoch
    dropouts = zeros(0,2);
    
    % No previous bidder
    previousBidder = missing;
    
    epoch = 1;
    
    while numel(players) > 1
        % Step 8:
        %   If there are 2 or more players, continue
        
        % Step 1:
        %   Set the new bid
        if numel(bids) == 0
            newBid = incr;
        else
            newBid = bids(end,2) + incr;
        end
        
        % Step 2:
        %   Select a player at random from players
        %   who are not the previous bidder
        otherPlayers = players(players ~= previousBidder);
        currentPlayer = otherPlayers(randi(numel(otherPlayers)));

        if isempty(bids)
            % Step 3:
            %   Place the first bid
            
            action = "bid";
        elseif rand(s) <= dropoutRate
            % Step 4:
            %   Generate random number and check against dropoutRate

            action = "dropout";
        else
            % Step 5:
            %   Calculate money gained by bidding
            idx = (bids(:,1) == currentPlayer);
            if any(idx)
                % Get the previous bid made by currentPlayer
                currentPlayerBids = bids(idx,2);
                previousBid = currentPlayerBids(end);
            else
                % No bids previously made by currentPlayer
                previousBid = 0;
            end
            gain = 1 - newBid;

            % Step 6:
            %   Calculate money lost by dropping out
            if epoch > 2 && previousBid == bids(end-1,2)
                % If the player made the second-to-last bid, they lose
                % money if they drop out
                loss = -previousBid;
            else
                % Otherwise, they do not lose any money
                loss = 0;
            end

            if gain > loss
                action = "bid";
            else
                action = "dropout";
            end
        end
        
        switch action
            case "bid"
                % Step 7:
                %   Record the bid and update the previous bidder
                bids = [bids; [currentPlayer newBid epoch]];
                previousBidder = currentPlayer;

                % Increment turn counter
                epoch = epoch + 1;
            case "dropout"
                % Step 8:
                %   Record the drop out and update the players
                dropouts = [dropouts; [currentPlayer epoch]];
                players(players == currentPlayer) = [];
        end
    end
    
    % Convert to tables
    bids = array2table(bids,'VariableNames',{'Player','Bid','Epoch'});
    dropouts = array2table(dropouts,'VariableNames',{'Player','Epoch'});
end
